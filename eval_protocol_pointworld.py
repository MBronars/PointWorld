#!/usr/bin/env python3
"""
eval_protocol_pointworld.py — Score a PointWorld checkpoint with the pwm-data-v1
evaluation protocol (pwm_final_paper/EVAL.md), by the protocol's own code.

WHAT IS PWM'S AND WHAT IS POINTWORLD'S
    Everything that defines a number is pwm_final_paper's, run unmodified: the
    frozen case lists and their selection, the evaluator (scripts/unified/
    eval_unified.py), its rollout loop (inference.windowed_rollout: 16-frame
    windows, context = the model's own committed output after the first window,
    ground truth entering only as the actuator's actions, the final window
    shortened to the clip's end), the scorers (TrajectoryScores: Euclidean, MSE,
    Chamfer, moving/still rows, passive-body pose), the trivial references, the
    receipts, the saved-predictions layout and SUMMARY.json / SUMMARY.md
    (scripts/unified/eval_protocol.plan / summarize / render).

    PointWorld replaces exactly one function: inference.run_diffusion_inference_df,
    the per-window predictor. Given a window batch it returns the window's
    predicted displacements in the form the evaluator integrates.

THE PREDICTOR, PER WINDOW
    PointWorld sees 1 context frame and predicts 10 (its release horizon). A
    16-frame window is covered by two PointWorld calls, 6 frames then 10, the
    second anchored on PointWorld's own 6th frame. That order keeps all 10 of
    every call's gripper frames inside the window, so no gripper frame is ever
    invented. Only a window shorter than 10 frames — the last one of a rollout,
    at the clip's end, where no future exists — holds the gripper at its last
    given frame to fill PointWorld's input.

    Each call goes through PointWorld's own test-mode pipeline (the same decode,
    flow sample and transform chain its training and eval use). The scene's
    unknown future frames are filled with its anchor frame; PointWorld reads
    only frame 0 of the scene (plus the whole gripper trajectory), and --check
    asserts the prediction is unchanged when the true future is supplied instead.

    PointWorld's 1 mm grid sampling reorders the scene and keeps one point per
    voxel. Every original row is mapped back to the surviving point nearest its
    frame-0 position (asserted within the voxel diagonal), so rows that shared a
    voxel take its displacement from their own position; the count is logged per call.

    Actuator rows are returned as given (their true positions), as the protocol's
    "given actions stay given".

WHAT THE RECEIPT FIELDS MEAN HERE
    `df_schedule` is recorded as `full_sequence` (a required protocol field;
    PointWorld has no sampler and is deterministic, so the seed changes nothing).
    The model's configuration (window, data roots, ruler, articulation) is the PWM
    run this baseline is matched against: --recipe.

USAGE (pointworld-env; the PWM env's model stack is not needed)
    python eval_protocol_pointworld.py protocol \\
        --checkpoint /data/group_data/katefgroup/runs_pw_mbronars/three_pointworld/matched_step_020000.pt \\
        --datasets pickup,rl_push --out <dir>
    python eval_protocol_pointworld.py check --checkpoint <ckpt>     # the leak / mapping self-checks
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

PW_ROOT = Path(__file__).resolve().parent
PWM_ROOT = Path(os.environ.get("PWM_ROOT", "/data/user_data/mbronars/packages/pwm_final_paper"))
SHIMS = [PW_ROOT / "pwm_shims", PW_ROOT / "pwm_shims" / "py"]
DEFAULT_RECIPE = "/data/user_data/mbronars/launch_pwm_data_v1_three/configs/unet_three_ctxgen_100m.yaml"
PW_DOMAIN = {"yikai": "pickup_yikai_small", "pickup": "pickup_release", "rl_push": "pickup_rl_push", "dex": "pickup_dex"}
PW_HORIZON = 10          # the release default; a checkpoint's own --pred_horizon (its contract) wins
VOXEL_DIAGONAL_M = 0.001 * 3 ** 0.5
SCHEDULE = "full_sequence"


# ─────────────────────────────────────────────────────────────────────────────
# One process, two code bases
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap():
    """Make PWM's evaluator and PointWorld's model importable side by side.

    - PWM first on sys.path, so `utils` is PWM's package (the evaluator needs a
      dozen of its submodules); PointWorld's top-level utils.py is loaded under a
      private name and its names are grafted onto that package — the two share
      no name (checked), so neither side loses anything.
    - `diffusion` (PWM's diffusion model, which imports diffusers) is replaced by
      a placeholder: inference.py imports its class, and the protocol path here
      never builds a PWM model.
    - pytorch3d and omegaconf come from the project-local shims, appended LAST.
    """
    for p in (str(PW_ROOT), str(PWM_ROOT)):
        if p in sys.path:
            sys.path.remove(p)
    sys.path[:0] = [str(PWM_ROOT), str(PW_ROOT)]
    for s in SHIMS:
        if str(s) not in sys.path:
            sys.path.append(str(s))
    os.environ.setdefault("PWM_PYTORCH3D", str(SHIMS[0]))
    os.environ.setdefault("WANDB_MODE", "disabled")

    if "diffusion" not in sys.modules:
        stub = types.ModuleType("diffusion")

        class TrajectoryFlowMatching:                  # noqa: D401 — placeholder
            def __init__(self, *a, **k):
                raise RuntimeError("PWM diffusion is not available in the PointWorld protocol path")
        stub.TrajectoryFlowMatching = TrajectoryFlowMatching
        sys.modules["diffusion"] = stub

    import utils as pwm_utils                           # PWM's package
    if not hasattr(pwm_utils, "__path__"):
        raise ImportError(f"`utils` resolved to {pwm_utils.__file__}, not PWM's package")
    if "pointworld_utils" not in sys.modules:
        spec = importlib.util.spec_from_file_location("pointworld_utils", PW_ROOT / "utils.py")
        pw_utils = importlib.util.module_from_spec(spec)
        sys.modules["pointworld_utils"] = pw_utils
        spec.loader.exec_module(pw_utils)
        for name, value in vars(pw_utils).items():
            if name.startswith("__") or hasattr(pwm_utils, name):
                continue
            setattr(pwm_utils, name, value)


def resolved_recipe(recipe: str):
    """The PWM run's configuration as its checkpoints carry it: DEFAULTS merged with the recipe
    and the normalization resolved — utils/config.resolve_config, minus its model-section
    validation (which imports PWM's model package; the recipe passed it when the run launched).
    The evaluator reads the window, sampler steps, data roots and scales from this."""
    from omegaconf import OmegaConf
    from utils.config import DEFAULTS
    from data.normalization import resolve_normalization
    cfg = OmegaConf.merge(OmegaConf.create(DEFAULTS), OmegaConf.load(recipe))
    if cfg.model.get("context_len") is None:
        cfg.model.context_len = cfg.data.context_len
    spec = OmegaConf.to_container(cfg.normalization, resolve=True)
    cfg.normalization = resolve_normalization(spec)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# The PointWorld model and its pipeline
# ─────────────────────────────────────────────────────────────────────────────

class PointWorldPredictor:
    """What the evaluator holds as its "model": PointWorld plus the PWM run's cfg."""

    def __init__(self, checkpoint: str, recipe: str, device):
        from omegaconf import OmegaConf
        self.recipe = recipe
        self.cfg_full = resolved_recipe(recipe)
        # The model config the evaluator consults describes the MODEL: PointWorld predicts particles
        # only, so no rigid-pose track is carried between windows (pose_native / predict_body_poses
        # off). How the DATA is read for the pose metrics stays the PWM run's (articulated_bodies).
        self.cfg = OmegaConf.merge(self.cfg_full.model, {"pose_native": False, "predict_body_poses": False})
        self.cfg_full.model = self.cfg
        self.device = device
        self.checkpoint = checkpoint
        self.domain = None                             # the PWM domain being scored, set by the caller
        self.stats = {"calls": 0, "rows": 0, "merged_rows": 0, "max_match_mm": 0.0, "held_gripper_frames": 0}
        self._build(checkpoint)

    def _build(self, checkpoint):
        """PointWorld's own Tester, on the checkpoint's own domains, horizon and statistics."""
        import torch
        from arguments import parse_args
        from evaluation.tester import Tester
        from pointworld.checkpoint_contract import read_checkpoint_contract, train_domains_from_data_contract
        from pwm_stream import PW_TO_PWM
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model_contract, data_contract = read_checkpoint_contract(ck, context=checkpoint)
        domains = train_domains_from_data_contract(data_contract, context=checkpoint)
        # A checkpoint trained with the binary gripper label has a 10-wide gripper input (9 without).
        self.gripper_open = int(ck["model"]["robot_proj.fc1.weight"].shape[1]) == 10
        rfeat = "robot_flows,gripper_open,robot_velocity,robot_acceleration" if self.gripper_open \
            else "robot_flows,robot_velocity,robot_acceleration"
        sfeat = "scene_flows,gripper_open,dist2robot" if self.gripper_open else "scene_flows,dist2robot"
        del ck
        argv = ["eval", "--domains", ",".join(domains),
                "--data_dirs", ",".join(f"pwm://{PW_TO_PWM[d]}@{self.recipe}" for d in domains),
                "--norm_stats_path", model_contract["norm_stats_path"],
                "--robot_features", rfeat,
                "--scene_features", sfeat,
                "--batch_size", "8", "--num_workers", "1", "--model_path", checkpoint,
                "--eval_skip_viz", "true", "--allow_missing_confidence_mask", "true"]
        old, sys.argv = sys.argv, argv
        try:
            args = parse_args()
        finally:
            sys.argv = old
        self.tester = Tester(args)
        self.model = self.tester.model.eval()
        self.args = self.tester.args
        self.horizon = int(getattr(self.args, "pred_horizon", PW_HORIZON) or PW_HORIZON)
        print(f"[pointworld] {checkpoint}: domains {domains}, horizon {self.horizon}", flush=True)

    def eval(self):
        self.model.eval()
        return self

    # --- one PointWorld call: anchor scene + 11 gripper frames -> 10 scene frames
    def _pipeline(self, scene0, robot, scene_future=None):
        import numpy as np
        from pwm_stream import FluidIterable, clip_fields, pickup_gripper_open
        from dataset_components.dataloader import _apply_sample_pipeline
        from dataset_components.collate import custom_collate_fn
        T = self.horizon + 1
        if scene_future is None:
            scene = np.repeat(scene0[None], T, axis=0)
        else:
            scene = np.concatenate([scene0[None], scene_future], axis=0)
        shift = np.concatenate([scene0, robot[0]], axis=0).mean(axis=0).astype(np.float32)
        sample = clip_fields((scene - shift).astype(np.float32), (robot - shift).astype(np.float32),
                             gripper_open=pickup_gripper_open(robot) if self.gripper_open else None)
        pw_domain = PW_DOMAIN[self.domain]
        ds = _apply_sample_pipeline(FluidIterable([sample]), f"pwm-eval://{self.domain}", pw_domain, "test",
                                    self.args, 0, False, None)
        items = list(ds)
        if len(items) != 1:
            raise RuntimeError(f"PointWorld's pipeline dropped the window ({len(items)} samples out)")
        batch = custom_collate_fn(items, self.args)
        return batch, shift

    def _call(self, scene0, robot, scene_future=None):
        """scene0 (Ns,3) m, robot (1+H,Nr,3) m -> predicted scene (H,Ns,3) m, row order of scene0."""
        import numpy as np
        import torch
        from scipy.spatial import cKDTree
        batch, shift = self._pipeline(scene0, robot, scene_future)
        gpu = {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            out = self.model(gpu, training=False)
        pred = out["scene_flows"][0].float().cpu().numpy()                 # (11, Ns', 3), pipeline frame
        exists = batch["scene_exists"][0, 0].numpy().astype(bool)
        pipe_shift = np.asarray(batch["__shift_amount__"][0], dtype=np.float32).reshape(-1)[:3] \
            if "__shift_amount__" in batch else np.zeros(3, np.float32)
        # __shift_amount__ holds MINUS the pipeline's accumulated shift (center_shift runs twice in test
        # mode, the second time after grid sampling), so undoing it subtracts it.
        pred = pred[:, exists] - pipe_shift + shift                        # back to the evaluator's frame
        kept0 = batch["scene_flows"][0, 0].float().numpy()[exists] - pipe_shift + shift
        dist, idx = cKDTree(kept0).query(scene0)
        self.stats["calls"] += 1
        self.stats["rows"] += len(scene0)
        self.stats["merged_rows"] += int(len(scene0) - len(np.unique(idx)))
        self.stats["max_match_mm"] = max(self.stats["max_match_mm"], float(dist.max()) * 1000.0)
        if dist.max() > VOXEL_DIAGONAL_M + 1e-6:
            dump = os.environ.get("PW_DEBUG_DUMP")
            if dump:
                np.savez(dump, scene0=scene0, robot=robot, kept0=kept0, dist=dist, idx=idx)
            raise AssertionError(f"row mapping off by {dist.max()*1000:.3f} mm (> voxel diagonal)")
        # Each row keeps its own anchor position and takes the matched point's displacement, so rows
        # that shared a voxel stay distinct instead of collapsing onto the surviving point.
        return scene0[None] + (pred[1:, idx] - kept0[idx][None])           # (10, Ns, 3)

    # --- the evaluator's per-window predictor
    def predict_window(self, win):
        """Window batch -> predicted future positions [B, F, N, 3] (metres)."""
        import numpy as np
        import torch
        C = win.context_len
        pos = win.positions.float().cpu().numpy()                          # [B, C+F, N, 3]
        F = pos.shape[1] - C
        valid = win.get_valid_mask().cpu().numpy()
        act = win.action_mask.cpu().numpy()
        out = pos[:, C:].copy()                                            # actuator rows stay given
        for b in range(pos.shape[0]):
            rrows = valid[b] & act[b]
            srows = valid[b] & ~act[b]
            # two-step indexing: a scalar, a slice and a boolean mask together move the mask's axis first
            robot_all = pos[b, C - 1:][:, rrows]                           # (1+F, Nr, 3): anchor + F given frames
            scene = pos[b, C - 1, srows]                                   # anchor (committed)
            H = self.horizon
            chunks = [F] if F <= H else [F - H * ((F - 1) // H)] + [H] * ((F - 1) // H)
            done = 0
            for keep in chunks:
                robot = robot_all[done: done + H + 1]
                if robot.shape[0] < H + 1:                                 # clip end: hold the last given frame
                    pad = H + 1 - robot.shape[0]
                    self.stats["held_gripper_frames"] += pad
                    robot = np.concatenate([robot, np.repeat(robot[-1:], pad, axis=0)], axis=0)
                pred = self._call(scene, robot)
                out[b, done: done + keep][:, srows] = pred[:keep]
                scene = pred[keep - 1]
                done += keep
        return torch.from_numpy(out).to(win.positions.device, win.positions.dtype)


def install(predictor: PointWorldPredictor):
    """Route the evaluator's predictor and loader to PointWorld. Nothing else changes."""
    import torch
    import inference
    import scripts.unified.eval_unified as EU

    def run_diffusion_inference_df(model, batch, diffusion=None, **kwargs):
        pred = predictor.predict_window(batch)
        deltas = pred - torch.cat([batch.positions[:, batch.context_len - 1: batch.context_len], pred[:, :-1]], dim=1)
        x0 = batch.prediction_stats.normalize(deltas, clip=False)
        back = batch.predictions_to_positions(batch.prediction_stats.denormalize(x0))
        err = (back - pred).abs().max().item()
        if err > 1e-5:
            raise AssertionError(f"delta round trip off by {err*1000:.4f} mm")
        return x0

    def load_model_and_diffusion(checkpoint, device):
        return predictor, None, predictor.cfg_full, int(predictor.cfg_full.flow_matching.get("inference_steps", 25))

    inference.run_diffusion_inference_df = run_diffusion_inference_df
    EU.run_diffusion_inference_df = run_diffusion_inference_df
    inference.load_model_and_diffusion = load_model_and_diffusion
    EU.load_model_and_diffusion = load_model_and_diffusion
    return EU


# ─────────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_unified(argv):
    """One eval_unified.py invocation, with PointWorld as the predictor."""
    import torch
    cap = float(os.environ.get("PW_MEM_FRACTION", "0"))
    if cap and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(cap, 0)   # share a GPU without squeezing its other job
    # The evaluator reads clips in this process. torch's default (every core of the node) oversubscribes
    # the job's CPUs; one thread starves PointWorld's own CPU preprocessing (437 s vs 125 s on pickup
    # windows). The job's own CPU count avoids both.
    torch.set_num_threads(max(1, len(os.sched_getaffinity(0))))
    bootstrap()
    recipe = os.environ.get("PW_RECIPE", DEFAULT_RECIPE)
    ckpt = argv[argv.index("--checkpoint") + 1]
    domain = argv[argv.index("--domains") + 1]
    predictor = PointWorldPredictor(ckpt, recipe, torch.device("cuda"))
    predictor.domain = domain
    EU = install(predictor)
    os.chdir(PWM_ROOT)
    sys.argv = ["eval_unified.py", *argv]
    t0 = time.time()
    EU.main()
    s = predictor.stats
    print(f"[pointworld] {s['calls']} PointWorld calls in {time.time()-t0:.0f}s; rows mapped {s['rows']}, "
          f"sharing a voxel {s['merged_rows']} ({100*s['merged_rows']/max(1,s['rows']):.2f}%), "
          f"max match {s['max_match_mm']:.3f} mm; held gripper frames {s['held_gripper_frames']}", flush=True)


def cmd_protocol(args):
    bootstrap()
    import scripts.unified.eval_protocol as EP
    from utils.reproducibility import code_revision, file_sha256
    cfg = resolved_recipe(args.recipe)
    checkpoint = Path(args.checkpoint).resolve()
    out_dir = Path(args.out)
    only = [d for d in args.datasets.split(",") if d] if args.datasets else None
    sets = [s for s in args.sets.split(",") if s] if args.sets else None
    entries = EP.plan(checkpoint, EP.checkpoint_datasets(cfg), out_dir, SCHEDULE,
                      only_datasets=only, only_sets=sets, predictions=args.predictions)
    for e in entries:                                   # same words, this script as the evaluator
        i = e["words"].index("scripts/unified/eval_unified.py")
        e["words"] = [sys.executable, str(Path(__file__).resolve()), "unified", *e["words"][i + 1:]]
    if args.dry_run:
        for e in entries:
            print(" ".join(e["words"]))
        return
    ck_sha = file_sha256(str(checkpoint))
    env = {**os.environ, "PW_RECIPE": args.recipe, "PYTHONPATH": ""}
    for n, e in enumerate(entries, 1):
        receipt = Path(e["receipt"])
        label = f"[{n}/{len(entries)}] {e['dataset']} {e['case_set']}"
        if args.resume and EP._matches(receipt, e, ck_sha, file_sha256(e["manifest"]), SCHEDULE):
            print(f"{label}: kept {receipt}", flush=True)
            continue
        receipt.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"{label}: scoring ...", flush=True)
        with open(receipt.with_suffix(".log"), "w") as log:
            done = subprocess.run(e["words"], cwd=PWM_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if done.returncode:
            raise SystemExit(f"{label}: failed, see {receipt.with_suffix('.log')}")
        print(f"{label}: done in {(time.time()-t0)/60:.1f} min", flush=True)
    summary = dict(checkpoint=str(checkpoint), checkpoint_sha256=ck_sha, reported_checkpoint=False,
                   code=code_revision().get("git_commit", "")[:7], seed=0, schedule=SCHEDULE,
                   checkpoint_schedule=SCHEDULE, partial=bool(only or sets), report_folders=False,
                   method="PointWorld (eval_protocol_pointworld.py)", recipe=args.recipe,
                   sets=EP.summarize(entries))
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    (out_dir / "SUMMARY.md").write_text(EP.render(summary))
    print(f"wrote {out_dir / 'SUMMARY.md'}")
    if any(not r["reference_matches_manifest"] for r in summary["sets"]):
        raise SystemExit("No-motion number differs from a frozen list: the clips read are not the frozen ones")


def cmd_check(args):
    """Self-checks on real case windows: no future-scene leak, row mapping, round trip."""
    import numpy as np
    import torch
    bootstrap()
    predictor = PointWorldPredictor(args.checkpoint, args.recipe, torch.device("cuda"))
    install(predictor)
    import scripts.unified.eval_unified as EU
    from utils.reproducibility import seed_for_device
    cfg = resolved_recipe(args.recipe)
    H = predictor.horizon
    for domain in args.check_domains.split(","):
        predictor.domain = domain
        ds = EU.evaluation_dataset(cfg, domain, int(cfg.data.context_len), int(cfg.data.future_len))
        EU._prepare(ds, predictor, domain)
        from utils.evaluation import iter_clips
        worst = same = 0.0
        for i, (batch, _) in enumerate(iter_clips(ds, 3)):
            C = batch.context_len
            pos = batch.positions.float().numpy()
            valid, act = batch.get_valid_mask().numpy()[0], batch.action_mask.numpy()[0]
            s, r = valid & ~act, valid & act
            robot = pos[0, C - 1:C + H][:, r]
            # PTv3 reshuffles its serialization orders with torch.randperm on every forward, eval
            # included, so PointWorld is only reproducible under a fixed seed — as the protocol runs
            # it (eval_unified reseeds per clip). Each call here gets the same seed.
            seed_for_device(0, predictor.device); a = predictor._call(pos[0, C - 1, s], robot)
            seed_for_device(0, predictor.device); a2 = predictor._call(pos[0, C - 1, s], robot)
            seed_for_device(0, predictor.device)
            b = predictor._call(pos[0, C - 1, s], robot, scene_future=pos[0, C:C + H][:, s])
            same = max(same, float(np.abs(a - a2).max()))
            worst = max(worst, float(np.abs(a - b).max()))
        print(f"{domain}: same seed, same input: max diff {same*1000:.4f} mm | same seed, true future supplied: "
              f"max diff {worst*1000:.4f} mm")
        assert same < 1e-4, "PointWorld is not reproducible under a fixed seed"
        assert worst < 1e-4, "PointWorld's prediction depends on the scene's future frames"
    s = predictor.stats
    print(f"mapping: {s['rows']} rows, {s['merged_rows']} sharing a voxel, max match {s['max_match_mm']:.3f} mm")
    print("check OK")


def main():
    import argparse
    if len(sys.argv) > 1 and sys.argv[1] == "unified":
        return cmd_unified(sys.argv[2:])
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("protocol", "check"):
        q = sub.add_parser(name)
        q.add_argument("--checkpoint", required=True)
        q.add_argument("--recipe", default=DEFAULT_RECIPE)
        if name == "check":
            q.add_argument("--check_domains", default="pickup,rl_push")
        if name == "protocol":
            q.add_argument("--out", required=True)
            q.add_argument("--datasets", default=None)
            q.add_argument("--sets", default=None)
            q.add_argument("--predictions", default="headline", choices=("headline", "all", "none"))
            q.add_argument("--resume", action="store_true")
            q.add_argument("--dry_run", action="store_true")
    a = p.parse_args()
    {"protocol": cmd_protocol, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
