#!/usr/bin/env python3
"""
pwm_stream.py — Train PointWorld directly on the pwm-data-v1 corpus, with no
intermediate copy of the data.

A `--data_dirs` entry of the form

    pwm://<pwm domain>@<recipe.yaml>

is not a directory of shards: it is the PWM unified loader, driven with that
recipe's own data.roots, producing windows on the fly. Each window is handed to
PointWorld in exactly the byte layout a shard would have carried, so everything
downstream of `decode_data` — the flow sample, the camera sampling, the whole
transform pipeline — is the same code on the same bytes as the dumped-shard path
(pwm_three_to_pointworld_wds.py, which remains the way to freeze a corpus).

WHAT THIS BUYS
    No 81 GB copy, and the corpus cannot drift from the PWM run's: the exclusion
    manifests, pickup's shift_m, yikai_small's per-episode table height and
    rl_push's load-time point construction are read live from the same code the
    PWM run uses. Windows are drawn fresh every epoch instead of being frozen at
    dump time, so the baseline is not restricted to a fixed window sample.

WHAT IT COSTS
    The PWM loader must import inside the PointWorld env. Measured: pickup and
    rl_push import and run there unchanged; yikai's native adapter reaches
    `utils/pose_ops.so3_log`, which imports pytorch3d — for pose channels this
    script never reads. pointworld-env has no pytorch3d, so PWM_PYTORCH3D (see
    below) puts one on sys.path. Throughput is not a concern: one reader is
    ~59 windows/s against ~9 windows/s of training at batch 8.

ENV
    PWM_ROOT        the PWM checkout to import (default: packages/pwm_final_paper)
    PWM_PYTORCH3D   a directory holding a pytorch3d package to expose to the PWM
                    import, appended to the END of sys.path so the local torch
                    always wins. Only `matrix_to_quaternion` and
                    `quaternion_to_axis_angle` are reached, both pure tensor code.
                    Unset it once pointworld-env has its own pytorch3d.
    PWM_RELEASE_FROM / PWM_RELEASE_TO
                    a moved copy of the release (e.g. pwm-data-v1-compact on PSC): every recipe path under
                    PWM_RELEASE_FROM is re-pointed to PWM_RELEASE_TO, as gpwm's data.release_from /
                    data.release_to do (utils/config.py relocate_release; here applied to the whole recipe, so the
                    normalization file moves too). Both or neither; the recipe file itself is not edited.
"""
from __future__ import annotations

import io
import os
import sys
import warnings

import numpy as np

PWM_PREFIX = "pwm://"
PWM_ROOT = os.environ.get("PWM_ROOT", "/data/user_data/mbronars/packages/pwm_final_paper")

# PointWorld's release horizon: 1 context + 10 prediction frames. A run with --pred_horizon H
# streams 1 + H frames instead (build_pwm_source reads it off the args).
CLIP_LEN = 11

# The PWM domain behind each PointWorld domain name. The PointWorld name contains
# "pickup" because PointWorld dispatches on that substring in four places, and the
# pickup path (gripper points in the sample, no URDF sampler, no cameras) is the
# one these windows want.
PW_TO_PWM = {
    "pickup_yikai_small": "yikai",
    "pickup_release": "pickup",
    "pickup_rl_push": "rl_push",
    "pickup_dex": "dex",
    "pickup_folding": "folding",          # pwm-data-v1 folding_8mm (cloth, 30 Hz), 2026-10-22
    "pickup_pushing": "pushing",          # pwm-data-v1 pushing (block pushing: rod + block, 128 + 128 points), 2026-10-26
    "pickup_mould": "mould",              # pile manipulation (deformable_mould_new_6k_wds; needs a gpwm checkout with the mould domain), 2026-10-24
}

# Windows per nominal epoch, per domain. This sets ONLY bookkeeping — len(loader),
# and the epoch_count = sample_count / total_samples the trainer stops on. Sampling
# is unaffected: the train stream cycles episodes forever with fresh random starts.
# The numbers are the indexed clip counts TRAINING.md records for each corpus, so
# one nominal epoch is "one window per indexed clip".
NOMINAL_WINDOWS = {
    "train": {"yikai": 17661, "pickup": 42253, "rl_push": 50815, "dex": 16999, "folding": 4510, "mould": 5904, "pushing": 93184},
    "test": {"yikai": 334, "pickup": 200, "rl_push": 200, "dex": 174, "folding": 236, "mould": 96, "pushing": 100},
}

PWM_SPLIT = {"train": "train", "test": "val"}


# ─────────────────────────────────────────────────────────────────────────────
# Spec parsing
# ─────────────────────────────────────────────────────────────────────────────

def is_pwm_spec(data_dir: str) -> bool:
    return isinstance(data_dir, str) and data_dir.startswith(PWM_PREFIX)


def parse_pwm_spec(data_dir: str) -> tuple[str, str]:
    """'pwm://yikai@/path/recipe.yaml' -> ('yikai', '/path/recipe.yaml')."""
    body = data_dir[len(PWM_PREFIX):]
    if "@" not in body:
        raise ValueError(f"{data_dir!r} must be pwm://<domain>@<recipe.yaml>")
    domain, recipe = body.split("@", 1)
    if not os.path.isfile(recipe):
        raise FileNotFoundError(f"recipe {recipe} (from {data_dir!r}) does not exist")
    return domain, recipe


def nominal_windows(data_dir: str, split: str) -> int:
    domain, _ = parse_pwm_spec(data_dir)
    table = NOMINAL_WINDOWS[split if split in NOMINAL_WINDOWS else "train"]
    if domain not in table:
        raise KeyError(f"no nominal window count for PWM domain {domain!r}")
    return table[domain]


# ─────────────────────────────────────────────────────────────────────────────
# The PWM loader, imported lazily and inside the worker
# ─────────────────────────────────────────────────────────────────────────────

# The PWM modules `data/` reaches for under the name `utils`. All three are pure
# torch; none of them imports anything else from `utils` at module level.
PWM_UTILS_MODULES = ("pose_ops", "rigid_ops", "pose_native")


def _shim_pwm_utils() -> None:
    """Let PWM's `from utils.pose_ops import ...` work next to PointWorld's utils.py.

    PointWorld has a top-level *module* `utils.py`; PWM has a top-level *package*
    `utils/`. Whichever is imported first owns the name, and in a training process
    that is always PointWorld's — so PWM's import fails in the dataloader workers
    with "'utils' is not a package".

    Rather than fight over the name, load the three modules PWM's data path needs
    straight from their files, register them as `utils.<name>` and hang them off
    the module that owns `utils`. `from utils.pose_ops import so3_log` then finds
    `utils.pose_ops` in sys.modules and reads the attribute, and PointWorld's own
    `utils` keeps every name it had.
    """
    owner = sys.modules.get("utils")
    if owner is None or hasattr(owner, "__path__"):
        return          # nothing shadowing it, or `utils` is already PWM's package
    import importlib.util
    for name in PWM_UTILS_MODULES:
        full = f"utils.{name}"
        if full in sys.modules:
            continue
        path = os.path.join(PWM_ROOT, "utils", f"{name}.py")
        spec = importlib.util.spec_from_file_location(full, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path} as {full}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        spec.loader.exec_module(module)
        setattr(owner, name, module)


def _import_pwm():
    if PWM_ROOT not in sys.path:
        sys.path.insert(0, PWM_ROOT)
    shim = os.environ.get("PWM_PYTORCH3D")
    if shim and shim not in sys.path:
        # END of sys.path: the local torch and every local package keep priority.
        sys.path.append(shim)
    _shim_pwm_utils()
    from data import unified_dataset as U            # noqa: E402
    from data.unified_dataset import build_domain_dataset  # noqa: E402
    return U, build_domain_dataset


def _release_relocation():
    old, new = os.environ.get("PWM_RELEASE_FROM"), os.environ.get("PWM_RELEASE_TO")
    if not old and not new:
        return None
    if not (old and new) or not (os.path.isabs(old) and os.path.isabs(new)):
        raise ValueError("PWM_RELEASE_FROM and PWM_RELEASE_TO must both be set to absolute paths")
    return old.rstrip("/"), new.rstrip("/")


def _relocate(node, relocation):
    """Every string under the old release prefix, re-pointed (gpwm utils/config.py relocated_path)."""
    if isinstance(node, dict):
        return {k: _relocate(v, relocation) for k, v in node.items()}
    if isinstance(node, list):
        return [_relocate(v, relocation) for v in node]
    if isinstance(node, str):
        old, new = relocation
        return new + node[len(old):] if node == old or node.startswith(old + "/") else node
    return node


def _load_recipe(recipe: str) -> dict:
    import yaml
    with open(recipe) as f:
        cfg = yaml.safe_load(f)
    relocation = _release_relocation()
    return cfg if relocation is None else _relocate(cfg, relocation)


def build_pwm_dataset(domain: str, recipe: str, mode: str, batch_size: int = 4, clip_len: int = CLIP_LEN):
    """The PWM dataset for one domain, windowed to PointWorld's 1 + 10 frames.

    raw=True: metres, unnormalized — PointWorld applies its own statistics. The
    domain's coordinate shift is part of the adapter and still applies.
    """
    U, build_domain_dataset = _import_pwm()
    cfg = _load_recipe(recipe)
    roots = U.resolve_data_roots(cfg["data"]["roots"])
    if domain not in roots:
        raise ValueError(f"{domain} is not in {recipe}'s data.roots ({sorted(roots)})")
    training = mode == "train"
    ds = build_domain_dataset(domain, split=PWM_SPLIT[mode], training=training,
                              future_len=clip_len - 1, context_len=1,
                              normalization=cfg["normalization"], raw=True, roots=roots)
    ds.batch_size = batch_size
    # The recipe's data.window_motion, applied as PWM's train.py applies it: training starts drawn with
    # p(start) ∝ (mean predicted-particle step over the window's future frames + floor_mm)^alpha.
    weighting, _ = U.validate_window_sampling(cfg["data"].get("window_motion"))
    if training and domain in weighting:
        ds.motion_alpha = weighting[domain]["alpha"]
        ds.motion_floor_m = weighting[domain]["floor_mm"] / 1000.0
    return ds


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


# ── The binary gripper label (pickup) ────────────────────────────────────────
# The rule the Isaac replay commands the gripper by: closed when the mean finger joint
# q < 0.035 m (pickup_replay_isaac/settings/v1.json, guided_playback isaac_float.py). On the
# loader's points q = (finger-centroid gap - 27 mm) / 2, so closed is a gap under 97 mm.
# The pickup gripper is 671 rows in the loader's order: hand (241), then the two fingers.
GRIPPER_OPEN_MIN_GAP_M = 0.097
PICKUP_GRIPPER_ROWS = 671
PICKUP_FINGER_ROWS = (slice(241, 453), slice(453, 671))


def pickup_finger_gap(robot: np.ndarray) -> np.ndarray:
    """Finger-centroid gap per frame, metres: robot [T, 671, 3] in the loader's row order -> [T]."""
    robot = np.asarray(robot)
    if robot.ndim != 3 or robot.shape[1] != PICKUP_GRIPPER_ROWS:
        raise ValueError(f"the pickup gripper is {PICKUP_GRIPPER_ROWS} rows (hand, finger, finger); got {robot.shape}")
    a, b = PICKUP_FINGER_ROWS
    return np.linalg.norm(robot[:, a].mean(axis=1) - robot[:, b].mean(axis=1), axis=-1)


def pickup_gripper_open(robot: np.ndarray) -> np.ndarray:
    """gripper_open per frame, [T, 1] float32: 1 open, 0 closed (see GRIPPER_OPEN_MIN_GAP_M)."""
    return (pickup_finger_gap(robot) >= GRIPPER_OPEN_MIN_GAP_M).astype(np.float32)[:, None]


def wants_gripper_open(args) -> bool:
    """True when the run's feature lists ask for gripper_open."""
    return any("gripper_open" in (getattr(args, k, None) or ()) for k in ("robot_features", "scene_features"))


def clip_fields(scene: np.ndarray, robot: np.ndarray, gripper_open: np.ndarray | None = None) -> dict:
    """One PointWorld sample, in the byte layout the shards carry.

    Going through np.save keeps `decode_data` on its normal path, so a streamed
    sample and a dumped one are the same bytes by the time anything reads them.
    """
    zeros = np.zeros((scene.shape[0], 1), dtype=np.float32)
    opened = zeros if gripper_open is None else np.asarray(gripper_open, dtype=np.float32).reshape(scene.shape[0], 1)
    return {
        "view0_scene_flows.npy": _npy_bytes(scene),
        "view0_scene_visibility.npy": _npy_bytes(np.ones(scene.shape[:2], dtype=bool)),
        "robot_flows.npy": _npy_bytes(robot),
        "joint_positions.npy": _npy_bytes(zeros),
        "right_gripper_open.npy": _npy_bytes(opened),
        "left_gripper_open.npy": _npy_bytes(zeros),
    }


def iter_window_arrays(domain: str, recipe: str, mode: str, batch_size: int = 4, clip_len: int = CLIP_LEN,
                       gripper_open: bool = False):
    """Yield (scene [T,Ns,3], robot [T,Nr,3]) in metres, centered, valid rows only.

    gripper_open=True (pickup only) yields a third item, the binary label [T,1].

    Rows are split by the PWM batch's action_mask — True is the actuator, which
    becomes robot_flows; the rest is the scene, the rows the PWM protocol scores.
    Each window is centered on the mean of its own frame-0 points, as
    pickup_to_pointworld_wds.py centers a clip.

    The held-out stream cycles its single deterministic pass rather than ending,
    so evaluation can draw any number of batches; the windows repeat in a fixed
    order.
    """
    import torch
    from torch.utils.data import DataLoader

    if gripper_open and domain != "pickup":
        raise ValueError(f"gripper_open is defined for pickup only (the 671-row gripper); got domain {domain}")
    while True:
        ds = build_pwm_dataset(domain, recipe, mode, batch_size, clip_len)
        produced = 0
        for batch in DataLoader(ds, batch_size=None, num_workers=0):
            positions, action_mask = batch.positions, batch.action_mask
            valid = batch.get_valid_mask()
            if positions.shape[1] != clip_len:
                raise ValueError(f"expected {clip_len}-frame windows, got {positions.shape[1]}")
            for b in range(positions.shape[0]):
                v = valid[b]
                robot_rows, scene_rows = v & action_mask[b], v & ~action_mask[b]
                if not bool(robot_rows.any()) or not bool(scene_rows.any()):
                    continue
                robot = positions[b][:, robot_rows].float().numpy().astype(np.float32)
                scene = positions[b][:, scene_rows].float().numpy().astype(np.float32)
                shift = np.concatenate([scene[0], robot[0]], axis=0).mean(axis=0).astype(np.float32)
                if gripper_open:
                    yield scene - shift, robot - shift, pickup_gripper_open(robot)
                else:
                    yield scene - shift, robot - shift
                produced += 1
        if produced == 0:
            import torch.utils.data as _tud
            info = _tud.get_worker_info()
            where = f" (worker {info.id} of {info.num_workers})" if info else ""
            raise RuntimeError(
                f"PWM stream for {domain}/{mode} produced no windows{where}. The PWM datasets "
                "partition their records across workers, so this usually means more workers than "
                "the split has shards — pickup's test root, for instance, is 4 shards.")
        # A training dataset cycles internally and never reaches here.


def iter_windows(domain: str, recipe: str, mode: str, batch_size: int = 4, clip_len: int = CLIP_LEN,
                 gripper_open: bool = False):
    """Yield PointWorld sample dicts forever — iter_window_arrays in shard bytes."""
    for fields in iter_window_arrays(domain, recipe, mode, batch_size, clip_len, gripper_open=gripper_open):
        yield clip_fields(*fields)


# ─────────────────────────────────────────────────────────────────────────────
# The fluid interface the PointWorld pipeline chains onto
# ─────────────────────────────────────────────────────────────────────────────

try:
    from torch.utils.data import IterableDataset as _IterableBase
except ImportError:                                   # the dumper's env imports this module without torch
    _IterableBase = object


class FluidIterable(_IterableBase):
    """An iterable with webdataset's `.map(f, handler=...)`, and nothing else.

    A torch IterableDataset, so a single-domain run can hand it straight to the DataLoader
    (with several domains it sits inside wds.RandomMix, which is one already); otherwise the
    DataLoader takes it for a map-style dataset and asks for its length.

    dataset_components/pipeline.py and dataloader.py chain `.map` onto whatever
    build_dataset returns; that is the entire interface a source has to offer.
    Handler semantics follow webdataset: no handler re-raises, and a handler that
    returns True skips the sample.
    """

    def __init__(self, source):
        super().__init__()
        self._source = source

    def __iter__(self):
        return iter(self._source() if callable(self._source) else self._source)

    def map(self, f, handler=None):
        parent = self

        def mapped():
            for sample in parent:
                try:
                    out = f(sample)
                except Exception as exc:               # noqa: BLE001 — webdataset's own contract
                    if handler is None or not handler(exc):
                        raise
                    continue
                if out is not None:
                    yield out

        return FluidIterable(mapped)


def build_pwm_source(data_dir: str, mode: str, args, rank: int = 0) -> FluidIterable:
    """The source end of the PointWorld pipeline, reading PWM windows live.

    The PWM dataset is constructed inside `__iter__`, so it is built in the
    dataloader worker that will read it. PWM's own datasets partition their
    records by (rank, worker) through wds.utils.pytorch_worker_info, which is why
    the workers get disjoint windows without anything here saying so.
    """
    domain, recipe = parse_pwm_spec(data_dir)
    batch_size = max(1, int(getattr(args, "pwm_loader_batch", 4)))
    clip_len = 1 + int(getattr(args, "pred_horizon", CLIP_LEN - 1) or CLIP_LEN - 1)

    gripper_open = wants_gripper_open(args)

    def source():
        return iter_windows(domain, recipe, mode, batch_size, clip_len, gripper_open=gripper_open)

    if mode == "train" and getattr(args, "deterministic_train", False):
        warnings.warn("pwm:// streams draw random window starts; --deterministic_train "
                      "does not make the training stream reproducible")
    return FluidIterable(source)
