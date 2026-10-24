# PointWorld as a gpwm baseline: every change from upstream

This fork (`MBronars/PointWorld`, branch `gpwm-baselines`) trains NVlabs PointWorld as a baseline for **gpwm**, our
particle world model (`RobertRen1122/particle_world_model`; older names in file and run names say `pwm`, which is the
same model). Upstream is `NVlabs/PointWorld`. The branch starts at upstream `8dff83d` ("initial commit"), the commit
every run so far was trained from; upstream has one newer commit (`0548482`, dataset and checkpoint release: README,
environment files, `evaluation/tester.py`, `pointworld/checkpoint_contract.py`) that is not merged.

Rule for every change (user, 2026-10-22): keep PointWorld's released recipe; change only what our data or a fair
comparison forces, and say why.

## How PointWorld is trained here (current runs)

| | released PointWorld | our runs | why |
|---|---|---|---|
| data | its own webdataset shards | `pwm://<domain>@<recipe>`: windows streamed from the gpwm release through gpwm's own loader (`pwm_stream.py`) | same episodes, exclusions, train / held-out split and window policy as gpwm |
| window | 1 given + 10 predicted | 1 given + 16 predicted (`--pred_horizon 16`) | gpwm predicts 16 frames |
| frame rate | its data's | the data's native rate (20 fps picking, 30 fps folding) | same frames as gpwm |
| grid | as released | 1 mm (`--grid_size 0.001`); rows sharing a voxel keep their own position and take the voxel's predicted displacement at inference | our points are 3-9 mm apart; 1 mm merges none of them (0 of 1.32 M folding points) |
| scene features | position + normals + colours | position + distance to the gripper at every frame (+ `gripper_open` for picking) | the simulated data has no normals or colours |
| gripper features | position, velocity, acceleration (+ gripper state) | the same; picking adds the binary `gripper_open` (closed when the finger gap < 97 mm, the rule the simulated gripper is commanded by) | |
| sphere crop | training-time multi-sphere crop | off for new runs (`--sphere_crop false`; default `true` = released) | user decision 2026-10-22 |
| backbone | PTv3 "base" (410 M) | picking: released default; folding: 99 M "medium" (`--ptv3_size medium --predictor_dim 192`, small layout with wider channels) | user decision 2026-10-22 |
| optimizer | AdamW 1e-4 constant | unchanged | |
| budget | epochs | `--budget_windows` / matched snapshots so PointWorld sees the same windows and steps as the gpwm model it is compared with (folding: 8 windows/step x 200 k = 1.6 M) | equal training |
| window sampling | uniform | the recipe's `data.window_motion` where gpwm uses it (folding: alpha 1.0, floor 0.02 mm) | gpwm trains with it |

## Run log

| run | data | backbone | windows / steps | launcher |
|---|---|---|---|---|
| pointworld_pickup_open | pickup (pwm-data-v1) | released default | 32 x 200k (1.6M checkpoint = matched_step_050000) | `train_pointworld_pickup_open.sbatch` |
| pointworld_folding_medium | folding_8mm | medium 99M | 8 x 200k = 1.6M (= gpwm folding) | `train_pointworld_folding_medium.sbatch` |
| pointworld_mould_medium | pile (deformable_mould_new_6k_wds) through gpwm `mb/baselines` | medium 99M | 16 x 100k = 1.6M (= the action-conditioned windows of the pile gpwm run, 200k x 16 at 50 % dropout) | `train_pointworld_mould_medium.sbatch` |

## Changes to upstream files (commit "Library changes ...")

| file | change |
|---|---|
| `arguments.py` | `--pred_horizon`, `--grad_accum_steps`, `--budget_windows`, `--snapshot_matched_steps`, `--matched_windows_per_step`, `--robot_features` / `--scene_features` (feature lists incl. `gripper_open`, `dist2robot`), `--sphere_crop`, `--param_nan_checks`, `--include_gripper_in_scene` |
| `training/trainer.py` | gradient accumulation; stop on a window budget; matched-step snapshots counted in windows; `--param_nan_checks false` skips the two per-step whole-model parameter NaN scans (the loss NaN check stays; ~6 % of a folding step); `PW_NCCL_TIMEOUT_MIN` env sets the DDP collective timeout (default PyTorch's 10 min) |
| `training/checkpointing.py` | `latest.pt` written atomically (temp file + rename) with group-readable mode; `matched_step_NNNNNN.pt` snapshots; `model-last.pt` link for older scripts |
| `pointworld/checkpoint_contract.py`, `pointworld/base.py` | prediction horizon recorded in the checkpoint contract (old checkpoints default to 10); NaN diagnostics name the offending points |
| `ptv3/ptv3_arch.yaml` | `medium` layout (enc_channels [c, c, 192, 384, 640], enc_depths [2,2,2,6,2], heads [2,4,8,16,32]; dec [c, c, 192, 384]) |
| `dataset_components/dataloader.py`, `pipeline.py` | `pwm://` sources (worker count, nominal epoch length, held-out stream on one worker); `--sphere_crop` passes prob 0 |
| `dataset_components/transforms.py` | sphere crop returns the sample unchanged at prob 0; `merge_gripper_into_scene` (folding option) |
| `dataset_components/decoders.py`, `robot.py` | particle-based pushing / folding / pickup shards: gripper points read from `robot_flows` (no URDF) |
| `evaluation/tester.py`, `utils.py`, `visualization/prediction_viz/sample.py`, `scene_featurizer.py` | simulation domains need no real-robot confidence masks; folding placeholder robot entry for the visualizer; dead code commented |
| `environments/requirements.txt` | flash-attn line commented (installed from a prebuilt wheel, see PSC_SETUP.md) |

## Added files

- `pwm_stream.py`: the `pwm://` source. Env: `PWM_ROOT` (gpwm checkout), `PWM_PYTORCH3D` (a directory with a
  `pytorch3d` package for gpwm's import), `PWM_RELEASE_FROM` / `PWM_RELEASE_TO` (read a moved release copy, e.g. the
  compact release on PSC). Binary `gripper_open` for pickup.
- `fit_pwm_norm_stats.py` (+ `merge_norm_stats.py`, `recompute_norm_stats_with_grippers.py`): PointWorld's
  normalization statistics fitted on gpwm windows (20,000 windows, parallel workers, KD-tree distances).
- `psc/norm_stats/{folding,pickup_open}/norm_stats.json`: the statistics the current runs use.
- Converters to webdataset shards (`*_to_pointworld_wds.py`, earlier runs only).
- `eval_protocol_pointworld.py`, `infer_pointworld.py` (+ `INFERENCE.md`), planning baselines (`imagined_mppi.py`,
  `mppi_mould.py`, `replay_actions*.py`, `sweep/`), visualization scripts.
- `*.sbatch` (repo root): launchers on the CMU babel cluster. `psc/`: launchers for PSC Bridges-2 (`PSC_SETUP.md`).

## Checks

- 2026-10-09: rows sharing a 1 mm voxel no longer collapse at inference.
- 2026-10-22: `--sphere_crop false` keeps every point (far cluster test: 800/800 vs 400/800 with the crop).
- 2026-10-23: the pwm:// stream on the compact release (PSC) vs pwm-data-v1, 24 held-out windows each: folding
  within 0.000134 mm, pickup within 0.0024 mm (the release's stated worst cases), pickup gripper_open identical;
  gpwm's psc-port code reads the old release identically to the code the runs used.
