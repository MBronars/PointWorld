# Running a PointWorld checkpoint elsewhere

For someone who has `pwm_final_paper` and wants to run **inference** on a PointWorld checkpoint
we trained. No training, no evaluator, no PWM data loader.

Everything below was measured in the env this repo trains in (2026-09-23).

## What has to travel

1. **This code.** A checkpoint stores weights and a config, not the model, so the classes have to
   come from here. The inference path touches **33 files, 10,260 lines** — the PTv3 backbone
   (`ptv3/`), the model (`pointworld/`), the preprocessing transforms (`dataset_components/`) and
   `arguments.py` / `utils.py`. `infer_pointworld.py` is the entry point and deliberately imports
   none of `training/`, `evaluation/`, wandb, webdataset or the PWM loader.
2. **The checkpoint** (`matched_step_NNNNNN.pt` or `latest.pt`, 4.9 GB each).
3. **The normalization statistics folder** the run was trained with — the `norm_stats.json` the
   model divides by. The checkpoint's model contract records the path it had on OUR machine, which
   will not exist elsewhere, so copy the folder and pass `--norm_stats`. Ours:

   | run | statistics |
   |---|---|
   | three-domain (pickup + rl_push + yikai_small), horizon 10 | `/data/group_data/katefgroup/runs_pw_mbronars/three_stream/norm_stats/` |
   | dex, horizon 16 | `/data/group_data/katefgroup/runs_pw_mbronars/dex_stream/norm_stats/` |
   | two-domain (pickup + yikai_small_fs0), horizon 16 | `/data/group_data/katefgroup/runs_pw_mbronars/two_stream/norm_stats/` |

The prediction horizon is read from the checkpoint (`pred_horizon` in its model contract), so the
same command runs a 10-frame and a 16-frame model without being told which it is.

## The environment: a separate one

**Do not add these to a PWM env.** PointWorld's backbone needs four compiled packages, and two of
them (`torch_scatter`, `flash_attn`) are torch C++ extensions locked to an exact torch build. A
PWM env on a different torch cannot load them, and installing them without care lets pip replace
torch — which is how such an env breaks. They are not optional: this model instantiates **56
`spconv.SubMConv3d` layers holding 284.5M of its 409.8M parameters**, attention calls
`flash_attn_varlen_qkvpacked_func` with no fallback, and pooling makes 7 `torch_scatter` calls.

The env we train and run inference in, exactly:

```bash
conda create -n pointworld-infer python=3.10 -y && conda activate pointworld-infer
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install spconv-cu124==2.3.8 timm==1.0.19 scipy==1.11.2 numpy==1.26.4 \
            opencv-python==4.11.0.86 pyyaml==6.0.2 addict==2.4.0 einops==0.8.2
pip install torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
pip install flash_attn==2.7.4.post1 --no-build-isolation
```

- Match the CUDA build to the machine; `cu124` is what we use.
- `torch_scatter` and `flash_attn` must match the torch version. Off the wheel index they build
  from source, which needs `nvcc` and takes tens of minutes (flash-attn longer).
- `spconv` ships its own CUDA kernels and is the least torch-sensitive of the four.

**If a separate env is impossible**, the deps can go into a directory instead of an env:
`pip install --target <dir> --no-deps <package>`, then put `<dir>` on `PYTHONPATH`. `--no-deps` is
what stops pip from pulling another torch; deleting the directory undoes everything. The compiled
packages still have to match that env's torch, so this only works if wheels exist for it.

## Running it

```bash
python infer_pointworld.py --checkpoint <run>/matched_step_060000.pt \
    --norm_stats <the folder from the table> --selftest --report
```

`--selftest` runs a synthetic window and prints the per-frame motion, which is the quickest check
that the env is correct. On a real window:

```bash
python infer_pointworld.py --checkpoint ... --norm_stats ... --window window.npz --out pred.npz
```

`window.npz` holds `scene` (Ns, 3) — the scene points at the anchor frame — and `robot`
(1 + H, Nr, 3) — the actuator's points at the anchor frame and then every given frame, in metres.
The output `pred` is (H, Ns, 3), row for row with `scene`. As a library:

```python
from infer_pointworld import PointWorldInference
model = PointWorldInference(checkpoint, norm_stats, device="cuda")
pred = model.predict(scene0, robot)        # (H, Ns, 3) metres
```

Two properties of the model worth knowing when feeding it:

- **The actuator is given.** The model reads its whole trajectory across the window and never
  predicts it. The scene's future is not an input; only its anchor frame is.
- **Rows come back in the order they went in**, but the model voxel-downsamples the scene at
  `grid_size` (1 mm in our runs), so points sharing a voxel share a trajectory. `--report` prints
  how many rows that affected and the largest mapping distance.

## Checked on 2026-09-23

| checkpoint | horizon | result |
|---|---|---|
| `three_pointworld_ga4_cont/matched_step_060000.pt` | 10 | prediction (10, 512, 3); motion from the anchor 2.9 → 20.6 mm |
| `pointworld_dex/matched_step_010000.pt` | 16 | prediction (16, 512, 3); motion 0.6 → 9.3 mm |

Both on one A100, with row mapping exact (0.000 mm) on the synthetic window.

## If the environment is the blocker

Two ways around it, neither needing this env:

- **Predictions instead of a model.** The protocol run
  (`eval_protocol_pointworld.py`, see `pwm_final_paper/EVAL.md`) already writes
  `<case set>.predictions/<k>.npz` for every case of the headline set: predicted positions in
  metres, the rows they belong to, and which of them the model predicted. Enough to score or
  visualize our results without running anything.
- **Dropping the dependencies** would mean replacing `torch_scatter` (7 call sites, direct torch
  equivalents), flash attention (`scaled_dot_product_attention`, slightly different numerics) and
  `spconv` — the last being a submanifold-convolution reimplementation over 69% of the parameters,
  which is a rewrite, not a swap.
