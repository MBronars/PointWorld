#!/usr/bin/env python3
"""
pointworld_server.py

Loads a PointWorld checkpoint and serves inference over HTTP.
All PointWorld / training dependencies are isolated here.

Endpoints
---------
POST /predict
    Accepts: application/octet-stream  (a .npz blob)
    Required arrays in the npz:
        view0_scene_flows       (T, N_block, 3)  float32  centred block positions
        view0_scene_visibility  (T, N_block)      bool
        joint_positions         (T, 2)            float32
        right_gripper_pose      (T, 7)            float32  centred TCP
        right_gripper_open      (T, 1)            float32

    Optional arrays:
        view0_scene_normals     (T, N_block, 3)  int8     quantised normals
            (only required if the model was trained with scene_normals)

    Returns: application/octet-stream  (a .npz blob)
        scene_flows             (T, N_block, 3)  float32  predicted block positions (centred)
        (plus any other model output tensors)

GET /health
    Returns 200 with {"status": "ok", "model_path": "..."}.

Usage
-----
    python pointworld_server.py \
        --model_path  checkpoints/model-last.pt \
        --data_dir    data/push_pointworld \
        --port        5123

    # Then from the client:
    # curl -X POST http://localhost:5123/predict --data-binary @sample.npz
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import webdataset as wds
from flask import Flask, request, jsonify, Response

# ── Project imports (all PointWorld deps live on the server) ──────────────────
from arguments import parse_args as parse_train_args
from arguments import DOMAIN_TO_DATA_DIR
from dataset_components.dataloader import build_dataloader
from training.trainer import Trainer
from pointworld.checkpoint_contract import (
    apply_model_contract_to_args,
    read_checkpoint_contract,
    train_domains_from_data_contract,
)

# ── Monkey-patch grid_sample_transform to be an identity. ──────────────────
# rolling_infer_from_h5.py calls the model directly (no dataloader, no grid
# sieve). Even with grid_size=1e-6 (no voxel collapses), the upstream
# implementation reorders points by FNV hash, and the model's predictions are
# order-sensitive — predictions differ for the same set of points in a
# different order. Replacing with identity preserves point order, which is
# what the reference effectively gets.
import dataset_components.transforms as _ds_transforms
import dataset_components.pipeline as _ds_pipeline
def _identity_grid_sample(sample, grid_size=None, mode=None):
    return sample
_ds_transforms.grid_sample_transform = _identity_grid_sample
_ds_pipeline.grid_sample_transform   = _identity_grid_sample
print("[pointworld_server] grid_sample_transform → identity "
      "(preserves point order — required for rolling_infer_from_h5 parity)")

# ─────────────────────────────────────────────────────────────────────────────
# Globals (populated at startup)
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

_model      = None   # nn.Module
_device     = None   # torch.device
_train_args = None   # argparse.Namespace (fully resolved)
_max_batch_size = 2048  # set from --max_batch_size; bounds per-request batch_size
_norm_stats_dir = None  # Path to norm_stats/
_model_path_str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: serialise numpy dicts over the wire
# ─────────────────────────────────────────────────────────────────────────────

def npz_to_bytes(arrays: dict) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    buf.seek(0)
    return buf.read()


def bytes_to_npz(raw: bytes) -> dict:
    buf = io.BytesIO(raw)
    with np.load(buf, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


def npy_bytes(arr: np.ndarray) -> bytes:
    """Serialise a numpy array to .npy format (for WebDataset tar entries)."""
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Batched inference: write K windows as ONE shard, build the dataloader ONCE,
# run the model over every batch it yields, stack predictions back to (K, ...).
#
# This is the key throughput change: a planner replan that scores K candidate
# action sequences used to be K separate POSTs, each rebuilding the whole
# WebDataset → dataloader pipeline. Now it's one POST, one shard, one dataloader
# build, one (batched) model forward. Feature extraction / normalisation /
# collation still go through the real training dataloader — unchanged, just
# amortised over K windows instead of one shard per window.
# ─────────────────────────────────────────────────────────────────────────────

_SHARD_FIELDS = (
    "view0_scene_flows", "view0_scene_visibility",
    "joint_positions", "right_gripper_pose", "right_gripper_open",
)
_OPTIONAL_FIELDS = ("view0_scene_normals",)
_MIN_SHARD_SAMPLES = 128         # pad past gather_shard_paths' ~1MB tar filter
                                 # (50 was borderline for small-K requests without
                                 # scene_normals — ~21KB × 50 ≈ 1.05MB landed under
                                 # the threshold; 128 gives a comfortable margin)


def _split_request(arrays: dict) -> list:
    """
    Split request arrays into a list of K per-window dicts.

    Accepts batched arrays (leading K axis: view0_scene_flows is (K, T, N, 3))
    or a single window (view0_scene_flows is (T, N, 3)) for backward compat.
    """
    sf = arrays["view0_scene_flows"]
    batched = (sf.ndim == 4)
    K = int(sf.shape[0]) if batched else 1

    windows = []
    for i in range(K):
        w = {}
        for f in _SHARD_FIELDS:
            w[f] = arrays[f][i] if batched else arrays[f]
        for f in _OPTIONAL_FIELDS:
            if f in arrays:
                w[f] = arrays[f][i] if batched else arrays[f]
        windows.append(w)
    return windows


def _write_shard(windows: list, tmp_root: Path) -> int:
    """
    Write the K windows as one WebDataset shard. Returns n_total written.

    Real windows go first (keys test_00000000..test_{K-1}); if K is small the
    shard is padded by cycling the real windows so the tar clears the ~1MB
    discovery filter. Single shard + test mode = no shuffle, so the first K
    samples the dataloader yields are exactly the real windows in order.
    """
    K = len(windows)
    test_dir = tmp_root / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    writer = wds.TarWriter(str(test_dir / "test_00000.tar"))

    def fields_for(w):
        out = {f"{f}.npy": npy_bytes(w[f]) for f in _SHARD_FIELDS}
        for f in _OPTIONAL_FIELDS:
            if f in w:
                out[f"{f}.npy"] = npy_bytes(w[f])
        return out

    n_total = max(K, _MIN_SHARD_SAMPLES)
    for idx in range(n_total):
        writer.write({"__key__": f"test_{idx:08d}", **fields_for(windows[idx % K])})
    writer.close()

    meta = {"train": {"processed_count": 0}, "test": {"processed_count": n_total}}
    (tmp_root / "metadata_rank0.json").write_text(json.dumps(meta, indent=2))

    # norm_stats + sibling metadata jsons from the real dataset
    dst = tmp_root / "norm_stats"
    try:
        dst.symlink_to(_norm_stats_dir.resolve())
    except OSError:
        shutil.copytree(str(_norm_stats_dir), str(dst))
    _real_data_dir = Path(_train_args.data_dirs[0])
    for meta_file in _real_data_dir.glob("*.json"):
        target = tmp_root / meta_file.name
        if not target.exists():
            shutil.copy2(str(meta_file), str(target))

    return n_total


def _predict_batched(arrays: dict) -> dict:
    """
    K windows → one shard → one dataloader build → model forward over every
    batch the loader yields → tensor outputs stacked back to (K, ...) numpy.
    """
    windows = _split_request(arrays)
    K = len(windows)
    import time as _t; _T = {'t0': _t.time()}

    tmp_root = Path(tempfile.mkdtemp(prefix="pw_srv_"))
    try:
        n_total = _write_shard(windows, tmp_root)
        _T['shard'] = _t.time()
        # Per-request batch_size: fit the whole shard in one forward when
        # possible. Avoiding multi-batch splits also avoids the
        # torch.cat(chunks) step, which fails when the model emits
        # per-sample variable-size aux tensors (KNN-related debug outputs).
        target_bs = min(n_total, _max_batch_size)
        n_batches = -(-n_total // target_bs)   # ceil
        print(f"[predict] {K} window(s) → shard of {n_total} samples  "
              f"batch_size={target_bs}  ({n_batches} batch{'es' if n_batches != 1 else ''})")

        args = _make_dataloader_args(str(tmp_root), batch_size=target_bs)
        dl, _ = build_dataloader(
            args, mode="test", rank=0, world_size=1, override_splits="test",
        )
        if dl is None:
            raise RuntimeError(f"build_dataloader returned None (tmp={tmp_root})")
        _T['loader'] = _t.time(); _T['model'] = 0.0

        collected: dict = {}
        # `__shift_amount__` is the cumulative negative shift the dataloader
        # applied to the inputs via repeated center_shift calls (once shared,
        # once after grid_sample). To return scene_flows in the same frame the
        # client sent, we subtract it from the model's output (which lives in
        # the fully-centered frame) before responding.
        shift_chunks: list = []
        n_seen = 0
        for batch in dl:
            batch = {
                k: v.to(_device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            shift = batch.get("__shift_amount__")
            if isinstance(shift, torch.Tensor):
                shift_chunks.append(shift.detach().cpu())
            # DIAG: dump exact model inputs for side-by-side with reference.
            _diag_dir = os.environ.get("PW_DIAG_DUMP_DIR")
            if _diag_dir:
                os.makedirs(_diag_dir, exist_ok=True)
                _dump = {
                    k: v.detach().cpu().numpy()
                    for k, v in batch.items() if isinstance(v, torch.Tensor)
                }
                # Server may yield more samples than K (shard padding). Save
                # only the first K rows of each sample-axis tensor — those are
                # the real client windows.
                K_real = len(_split_request(arrays))
                _dump_trim = {
                    k: (v[:K_real] if hasattr(v, "shape") and len(v.shape) > 0 else v)
                    for k, v in _dump.items()
                }
                np.savez(os.path.join(_diag_dir, f"SRV_seen{n_seen:04d}.npz"),
                         **_dump_trim)
                print(f"  [DIAG SRV] seen={n_seen} K_real={K_real} "
                      f"scene_flows={tuple(batch['scene_flows'].shape)} "
                      f"scene_features={tuple(batch['scene_features'].shape)} "
                      f"robot_flows={tuple(batch['robot_flows'].shape)} "
                      f"robot_features={tuple(batch['robot_features'].shape)} "
                      f"shift_amount={batch.get('__shift_amount__')[:K_real].cpu().numpy().tolist() if isinstance(batch.get('__shift_amount__'), torch.Tensor) else None}")
            torch.cuda.synchronize(); _tm = _t.time()
            with torch.no_grad():
                out = _model(batch, training=False)
            torch.cuda.synchronize(); _T['model'] += _t.time() - _tm
            for k, v in out.items():
                if isinstance(v, torch.Tensor):
                    collected.setdefault(k, []).append(v.detach().cpu())
            any_t = next((v for v in out.values() if isinstance(v, torch.Tensor)), None)
            if any_t is not None:
                n_seen += any_t.shape[0]
            if n_seen >= K:
                break          # have all real windows; padding samples not needed

        if os.environ.get('PW_TIMING'): print(f"[timing] K={K} shard {_T['shard']-_T['t0']:.3f} loader_build {_T['loader']-_T['shard']:.3f} iterate_total {_t.time()-_T['loader']:.3f} model {_T['model']:.3f}", flush=True)
        if not collected:
            raise RuntimeError("Dataloader yielded no batches / model produced no tensors")

        result = {}
        skipped: list = []
        for k, chunks in collected.items():
            # Some model outputs are per-sample variable-size (KNN-related
            # debug tensors etc.) and can't be cat'd across batches. The
            # client only consumes `scene_flows`; everything else is best-
            # effort. Skip-and-warn instead of failing the whole request.
            try:
                stacked = torch.cat(chunks, dim=0)
            except RuntimeError as e:
                skipped.append((k, str(e)))
                continue
            if stacked.shape[0] < K:
                raise RuntimeError(
                    f"Got {stacked.shape[0]} predictions for '{k}', expected ≥ {K} "
                    f"— some windows were dropped by the dataloader."
                )
            result[k] = stacked[:K].numpy()

        if skipped and "scene_flows" not in result:
            # The one output we actually serve also failed to cat. Surface it.
            raise RuntimeError(
                "torch.cat failed for every model output, including scene_flows: "
                + "; ".join(f"{k}: {msg}" for k, msg in skipped)
            )
        if skipped:
            # Aux outputs failed but scene_flows is fine — log once per call.
            print(f"[predict] skipped {len(skipped)} aux output(s) with "
                  f"inconsistent batch shapes: {[k for k, _ in skipped]}")

        if shift_chunks and "scene_flows" in result:
            shifts = torch.cat(shift_chunks, dim=0)[:K].numpy()       # (K, 3)
            result["scene_flows"] = result["scene_flows"] - shifts[:, None, None, :]
        elif "scene_flows" in result:
            print("[predict] WARNING: no __shift_amount__ in batches — "
                  "scene_flows returned in dataloader-centered frame, "
                  "client uncentering will be off by the residual shift.")

        return result
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _make_dataloader_args(tmp_data_dir: str, batch_size: Optional[int] = None):
    """
    Return a copy of the training args with the eval domain pointing at
    the temporary shard directory, and the eval-mode flags set the same way
    the Tester sets them. Optionally override `batch_size` on the copy so
    the dataloader yields one big batch per /predict request instead of
    splitting into multiple training-sized batches.
    """
    import copy
    args = copy.deepcopy(_train_args)
    if batch_size is not None:
        args.batch_size = int(batch_size)

    DOMAIN = "pushing"

    # Replace the data_dir for every domain with tmp_data_dir so the
    # dataloader looks for shards there.
    for i, dom in enumerate(args.domains):
        if dom == DOMAIN:
            args.data_dirs[i] = tmp_data_dir
            break

    # ── Eval-mode flags that Tester.__init__ sets ────────────────────────
    # Without these, build_dataloader may skip shard discovery for test splits.
    train_domains = list(args.domains)
    args.train_domains_for_eval = train_domains
    args.eval_domain_whitelist  = train_domains  # evaluate all domains
    args._eval_override_active  = True

    # ── Match rolling_infer_from_h5.py observable behaviour ──────────────
    # The reference rollout calls model() directly on a batch built by
    # `build_window_batch` — no grid sieve, no point cap, and a specific
    # URDF sampling (n_hand=128, seed=42). Override the test-mode pipeline
    # to match that behaviour point-for-point:
    #   * grid_sample_transform: voxel-sieves the cloud at args.grid_size,
    #     killing points that share a voxel. Compounds across AR windows
    #     (each window's output is sieved again next call) — sparse frame
    #     0, sparser by frame 20.
    #   * enforce_max_num_points: caps scene N.
    #   * URDF robot sampler: dataloader presamples `max_robot_points`
    #     points with `seed=args.seed + rank`. Training had max=500 (the
    #     model is robust to N changes but the resulting centroid offset
    #     shifts the entire centered scene by ~1-2 cm, which is enough
    #     to corrupt the model's spatial priors).
    args.grid_size        = 1e-6     # << any reasonable cloud spacing
    args.max_scene_points = 10 ** 9  # >> any reasonable N_block
    args.max_robot_points = 128      # match rolling_infer_from_h5.py n_hand
    args.seed             = 42       # match rolling_infer_from_h5.py URDF seed

    # Ensure num_workers is set for eval (Tester copies from num_workers)
    if hasattr(args, "eval_num_workers"):
        args.eval_num_workers = getattr(args, "num_workers", 0)

    if os.environ.get('PW_NO_WORKERS'):                 # 2026-10-09: no per-request worker process (the shard is tiny; spawning one costs ~0.9 s a call)
        args.num_workers = 0
        if hasattr(args, 'eval_num_workers'): args.eval_num_workers = 0
    return args


# ─────────────────────────────────────────────────────────────────────────────
# Flask routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok", model_path=_model_path_str)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept a .npz blob with the shard arrays for ONE window (arrays shaped
    (T, ...)) or a BATCH of K windows (arrays shaped (K, T, ...)), run the
    dataloader → model pipeline once, return predictions as .npz. Tensor
    outputs come back with a leading K axis matching the request.
    """
    raw = request.get_data()
    if not raw:
        return jsonify(error="Empty request body"), 400

    _dq = os.environ.get('PW_DUMP_REQ')
    if _dq and not os.path.exists(_dq): open(_dq, 'wb').write(raw)
    try:
        arrays = bytes_to_npz(raw)
    except Exception as exc:
        return jsonify(error=f"Failed to parse npz: {exc}"), 400

    required = [
        "view0_scene_flows", "view0_scene_visibility",
        "joint_positions", "right_gripper_pose", "right_gripper_open",
    ]
    missing = [k for k in required if k not in arrays]
    if missing:
        return jsonify(error=f"Missing arrays: {missing}"), 400

    try:
        out_arrays = _predict_batched(arrays)
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[predict] error:\n{tb}")
        return jsonify(error=f"Predict error: {exc}", traceback=tb), 500

    resp_bytes = npz_to_bytes(out_arrays)
    return Response(resp_bytes, mimetype="application/octet-stream")


# ─────────────────────────────────────────────────────────────────────────────
# Startup: load model once
# ─────────────────────────────────────────────────────────────────────────────

def init_model(model_path: str, data_dir: str, extra_args: list[str] | None = None):
    """
    Load the PointWorld checkpoint and build the model.  Called once at server
    start.  Populates module-level globals used by the /predict endpoint.
    """
    global _model, _device, _train_args, _norm_stats_dir, _model_path_str
    _model_path_str = model_path

    os.environ["WANDB_MODE"] = "disabled"
    torch.set_float32_matmul_precision("high")

    # ── Synthetic argv for the Trainer arg parser ────────────────────────
    argv = [
        "--model_path", model_path,
        "--distributed", "false",
        "--eval_num_batches", "1",
        "--seed", "42",
    ]
    if extra_args:
        argv.extend(extra_args)

    prev_argv = sys.argv
    sys.argv = ["pointworld_server.py"] + argv
    try:
        args = parse_train_args()
    finally:
        sys.argv = prev_argv

    args.model_path  = model_path
    args.distributed = False

    # ── Read checkpoint contract ─────────────────────────────────────────
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model_contract, data_contract = read_checkpoint_contract(
        checkpoint, context="pointworld_server"
    )
    apply_model_contract_to_args(
        args, model_contract, context="pointworld_server",
        explicit_cli_dests=set(),
    )
    train_domains = train_domains_from_data_contract(
        data_contract, context="pointworld_server"
    )
    if not train_domains:
        raise RuntimeError("Checkpoint has no training domains.")

    # ── Resolve data dirs for every domain ───────────────────────────────
    args.domains  = list(train_domains)
    args.data_dirs = []
    for dom in args.domains:
        if dom in DOMAIN_TO_DATA_DIR:
            args.data_dirs.append(DOMAIN_TO_DATA_DIR[dom])
        else:
            args.data_dirs.append(data_dir)

    # ── Build model via Trainer ──────────────────────────────────────────
    trainer = Trainer(args, inference_only=True, data_info_dict=None)
    trainer.model.load_state_dict(checkpoint["model"], strict=False)
    trainer.model.eval()

    _model      = trainer.model
    _device     = trainer.device
    _train_args = args

    # ── Cache the RobotSampler across /predict calls ─────────────────────────
    # build_dataset() builds a fresh TorchRobotSampler (URDF parse + mesh
    # mappings) on every dataloader build — i.e. every /predict. It depends
    # only on (urdf, gripper_only, device), so memoise it. Patching the name
    # in the dataloader module is enough; build_dataset references it bare.
    import dataset_components.dataloader as _dl_mod
    _orig_robot_sampler_cls = _dl_mod.TorchRobotSampler
    _robot_sampler_cache: dict = {}

    def _cached_robot_sampler(*s_args, **s_kwargs):
        key = (s_kwargs.get("urdf_path"), s_kwargs.get("gripper_only"),
               s_kwargs.get("device"))
        if key not in _robot_sampler_cache:
            _robot_sampler_cache[key] = _orig_robot_sampler_cls(*s_args, **s_kwargs)
        return _robot_sampler_cache[key]

    _dl_mod.TorchRobotSampler = _cached_robot_sampler

    _norm_stats_dir = Path(data_dir) / "norm_stats"
    if not _norm_stats_dir.is_dir():
        raise FileNotFoundError(
            f"norm_stats/ not found in {data_dir}. "
            f"Make sure --data_dir points to the dataset produced by "
            f"h5_to_pointworld_wds.py."
        )

    print(f"\n{'═'*60}")
    print(f"  PointWorld server ready")
    print(f"  Model     : {model_path}")
    print(f"  Device    : {_device}")
    print(f"  Domains   : {args.domains}")
    print(f"  Norm stats: {_norm_stats_dir}")
    print(f"{'═'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_server_args():
    p = argparse.ArgumentParser(
        description="PointWorld inference server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_dir",   required=True,
                   help="Training dataset root (must contain norm_stats/)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5123)
    p.add_argument("--max_batch_size", type=int, default=2048,
                   help="Per-request batch_size cap. When K windows fit in "
                        "one batch (i.e. K <= max_batch_size), the request "
                        "runs as a single model forward and the torch.cat "
                        "across multiple batches (which can fail on aux "
                        "tensors with per-sample variable shapes) is avoided. "
                        "Raise as high as your GPU memory tolerates.")
    p.add_argument("--extra_args", nargs="*", default=[],
                   help="Extra flags forwarded to the Trainer arg parser")
    return p.parse_args()


if __name__ == "__main__":
    srv_args = parse_server_args()
    init_model(srv_args.model_path, srv_args.data_dir, srv_args.extra_args or None)
    _max_batch_size = int(srv_args.max_batch_size)
    print(f"[server] max_batch_size = {_max_batch_size}")
    app.run(host=srv_args.host, port=srv_args.port, threaded=False)