# Training the PointWorld baseline on PSC Bridges-2

Companion of gpwm's `docs/PSC_SETUP.md` (branch `mb/psc-port` of `RobertRen1122/particle_world_model`), which set up
the gpwm 500M run on PSC. What changed from upstream PointWorld and why: `GPWM_CHANGES.md`.

## What runs where

| item | PSC location |
|---|---|
| data: pwm-data-v1-compact (the 8 gpwm datasets, 258 GB) | `/ocean/projects/cis260367p/mbronars/projects/GPWM/data` (already copied for the gpwm run) |
| this repo, branch `gpwm-baselines` | `$PW_REPO`, e.g. `/ocean/projects/cis260367p/mbronars/projects/GPWM/baselines/PointWorld` |
| gpwm checkout, branch `mb/psc-port` (PointWorld imports gpwm's loader from it) | `$GPWM_REPO`; the gpwm run's own checkout works |
| run folders | `/ocean/projects/cis260367p/mbronars/projects/GPWM/trainings/pointworld/<exp>/` (family `pointworld`) |
| slurm logs | `/ocean/projects/cis260367p/mbronars/projects/GPWM/trainings/pointworld/slurm/` |

## 1. Code

```bash
git clone -b gpwm-baselines git@github.com:MBronars/PointWorld.git "$PW_REPO"
cd "$PW_REPO" && git remote add upstream https://github.com/NVlabs/PointWorld.git
```

## 2. Environment (`pointworld-env`; exact versions in `environments/pointworld-env-2026-10-23.freeze.txt`)

Python 3.10, torch 2.5.1 + CUDA 12.4. All of these support the L40S (sm_89) the runs use, and the H100 (sm_90).

```bash
conda create -n pointworld-env python=3.10 -y && conda activate pointworld-env
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install spconv-cu124==2.3.8 cumm-cu124==0.7.11
pip install torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
grep -v -E '^(torch|torchvision|spconv|cumm|torch_scatter|flash_attn)' environments/pointworld-env-2026-10-23.freeze.txt > /tmp/pw_rest.txt
pip install --no-deps -r /tmp/pw_rest.txt   # exact freeze (urdfpy pins networkx==2.2, the env runs 3.4.2); includes zstandard and webdataset
python -c "import torch, spconv, flash_attn, torch_scatter, zstandard; print(torch.cuda.get_arch_list())"
```

gpwm's loader imports `pytorch3d` on a path PointWorld never uses (yikai pose channels), so `pwm_stream.py` appends
`$PW_REPO/pwm_shims` to `sys.path` (env `PWM_PYTORCH3D`). Point the link at the gpwm environment on PSC (`gpwm`, built as `pwm211`
built by gpwm's PSC_SETUP.md); only two pure-tensor functions are reached:

```bash
ln -sfn "$(conda run -n gpwm python -c 'import pytorch3d, os; print(os.path.dirname(pytorch3d.__file__))')" "$PW_REPO/pwm_shims/pytorch3d"
```

## 3. Data

Nothing to convert. The launcher reads the compact release through gpwm's loader with
`PWM_RELEASE_FROM=<old pwm-data-v1 path> PWM_RELEASE_TO=/ocean/.../GPWM/data`: every path in the gpwm recipe under the
old release is re-pointed (as gpwm's `data.release_from/to`). Both runs use gpwm's recipe
`launch/psc_root/configs/unet_joint8_500m_clean4_muon.yaml` (sha256 1e308cbd...; its pickup block is identical to the
two-domain recipe the babel pickup run used, and the windows match exactly). Normalization statistics ship in
`psc/norm_stats/`.

Check before the first run (one held-out window):

```bash
cd "$PW_REPO"; export PWM_ROOT=$GPWM_REPO PWM_PYTORCH3D=$PW_REPO/pwm_shims \
  PWM_RELEASE_FROM=/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1 \
  PWM_RELEASE_TO=/ocean/projects/cis260367p/mbronars/projects/GPWM/data
python -c "import pwm_stream as S; r='$GPWM_REPO/launch/psc_root/configs/unet_joint8_500m_clean4_muon.yaml'; \
sc, rb = next(S.iter_window_arrays('folding', r, 'test', 4, 17)); print(sc.shape, rb.shape)"
```

## 4. Launch

`psc/train_pointworld.psc.sbatch` holds both current runs (`RUN=folding_medium`, `RUN=pickup_open`) with the
settings of their babel launchers. **Confirm the scheduler lines first**: account `cis260367p`, partition
`GPU-shared`, `--qos=gpu` and `--gres=gpu:l40s-48:2` (gl001-003: 8 L40S, 192 CPUs, 1 TB) are this script's assumptions (2 GPUs; batch per GPU and accumulation keep
the windows per step of the babel runs). Bridges-2 caps the wall time per job; resubmitting the same command resumes
from `<exp>/latest.pt`.

```bash
sbatch --job-name=pw_folding_medium --export=ALL,RUN=folding_medium psc/train_pointworld.psc.sbatch
sbatch --job-name=pw_pickup_open   --export=ALL,RUN=pickup_open   psc/train_pointworld.psc.sbatch
```

To continue a babel run instead of starting fresh, copy its `latest.pt` to `trainings/pointworld/<exp>/latest.pt`
(same GPU count and windows per step).

## Speed notes (babel, L40S)

Folding medium: ~9.7 windows/s on 2 L40S (1.6 M windows ~ 46 h). Partial compile, a bigger micro-batch (does not fit
at 8) and the NaN scans were measured on folding (2026-10-22): only removing the NaN scans helped (~6 %).
