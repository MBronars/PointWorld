# Dataset Spec — PointWorld Baseline on SIM1 Cloth Folding

Hand-off document for training a PointWorld baseline on the same data as ParticleDiffuser, so the comparison is apples-to-apples.

## TL;DR

- **Dataset root**: `/data/group_data/katefgroup/VLA/3DPCWM/robert_sim1/sim1_cloth_diverse_v1/`
- **Format**: per-trajectory `.npz` files (one file = one trajectory; T frames packed inside)
- **Total available**: 1245 trajectories, 1.2M frames across 8 categories
- **What to use**: **751 trajectories** from 4 active categories (see Section 1)
- **Held out**: **8 specific trajectories** (Section 2) — MUST NOT be in train
- **OOD eval traj** (for long-horizon rollout): `/data/group_data/katefgroup/VLA/3DPCWM/robert_sim1/sim1_cloth_variant_0002/pointcloud_dataset/000086.npz` (a different, older dataset — fully held out of training)

---

## 1. Categories — use these 4 only

The dataset has 8 categories but only 4 have meaningful cloth motion. Skip the other 4 (they have <0.15 mm/frame cloth motion = mostly static):

| category | dir | trajs | frames | filename pattern |
|---|---|---|---|---|
| `bimanual` | `bimanual/` | 150 | 355,054 | `existing_*.npz` |
| `single_arm` | `single_arm/` | 201 | 302,685 | `000###_000###_{left,right}.npz` |
| `ik_lift_random` | `ik_lift_random/` | 200 | 85,800 | `000###.npz` |
| `ik_push` | `ik_push/` | 200 | 81,800 | `000###.npz` |
| **TOTAL** | | **751** | **825,339** | |

**Skip**: `primitive_chain/`, `primitive_menu/`, `ik_tap/`, `primitive_random/` (low/no cloth motion).

---

## 2. Train/Val split — exact

### Validation (8 trajectories, DO NOT TRAIN ON)

```
/data/group_data/katefgroup/VLA/3DPCWM/robert_sim1/sim1_cloth_diverse_v1/bimanual/existing_000000.npz       # pants
/data/group_data/katefgroup/VLA/3DPCWM/robert_sim1/sim1_cloth_diverse_v1/bimanual/existing_000004.npz       # long_shirt
/data/group_data/katefgroup/VLA/3DPCWM/robert_sim1/sim1_cloth_diverse_v1/single_arm/000000_000000_left.npz  # short_shirt
/data/group_data/katefgroup/VLA/3DPCWM/robert_sim1/sim1_cloth_diverse_v1/single_arm/000001_000000_right.npz # short_shirt
/data/group_data/katefgroup/VLA/3DPCWM/robert_sim1/sim1_cloth_diverse_v1/ik_lift_random/000000.npz          # short_shirt
/data/group_data/katefgroup/VLA/3DPCWM/robert_sim1/sim1_cloth_diverse_v1/ik_lift_random/000001.npz          # short_shirt
/data/group_data/katefgroup/VLA/3DPCWM/robert_sim1/sim1_cloth_diverse_v1/ik_push/000000.npz                 # short_shirt
/data/group_data/katefgroup/VLA/3DPCWM/robert_sim1/sim1_cloth_diverse_v1/ik_push/000001.npz                 # short_shirt
```

**Coverage**: 2 trajs from each of 4 categories. Garment mix: 1 pants + 6 short_shirt + 1 long_shirt (proportional to training mix).

**Why this list**:
- Categories `single_arm`, `ik_lift_random`, `ik_push` contain **only short_shirt** trajs in the dataset (this is the dataset's structure, not a filter choice).
- `bimanual` has all 3 garments: 25 pants + 92 short_shirt + 33 long_shirt. We pick one pants (`existing_000000`) and one long_shirt (`existing_000004`) from there to cover the rare garment classes.

### Training (743 trajectories)

Everything else in the 4 categories: `751 - 8 = 743 trajs`.

Per-category training counts:

| category | train trajs |
|---|---|
| bimanual | 148 |
| single_arm | 199 |
| ik_lift_random | 198 |
| ik_push | 198 |
| **total** | **743** |

### Verification snippet

A short Python check the PointWorld team should run before training to confirm split correctness:

```python
import os, glob
ROOT = '/data/group_data/katefgroup/VLA/3DPCWM/robert_sim1/sim1_cloth_diverse_v1'
CATEGORIES = ['bimanual', 'single_arm', 'ik_lift_random', 'ik_push']
VAL = {
    'bimanual/existing_000000.npz',
    'bimanual/existing_000004.npz',
    'single_arm/000000_000000_left.npz',
    'single_arm/000001_000000_right.npz',
    'ik_lift_random/000000.npz',
    'ik_lift_random/000001.npz',
    'ik_push/000000.npz',
    'ik_push/000001.npz',
}
all_files = []
for cat in CATEGORIES:
    all_files.extend(sorted(glob.glob(f'{ROOT}/{cat}/*.npz')))
val_resolved = [f for f in all_files if os.path.relpath(f, ROOT) in VAL]
train_resolved = [f for f in all_files if os.path.relpath(f, ROOT) not in VAL]
assert len(all_files) == 751,  f"expected 751 total, got {len(all_files)}"
assert len(val_resolved) == 8, f"expected 8 val, got {len(val_resolved)}"
assert len(train_resolved) == 743
assert set(train_resolved) & set(val_resolved) == set()
print(f"OK: {len(train_resolved)} train, {len(val_resolved)} val, disjoint")
```

---

## 3. .npz file schema (per trajectory)

Each `.npz` file is one trajectory. T frames of bimanual robot folding cloth on a table.

| key | shape | dtype | meaning |
|---|---|---|---|
| `cloth_points` | `[T, N_cloth, 3]` | float32 | per-frame cloth vertex positions (world frame, meters) |
| `left_ee_points` | `[T, 1000, 3]` | float32 | left gripper hardware point cloud, per frame |
| `right_ee_points` | `[T, 1000, 3]` | float32 | right gripper point cloud |
| `ee_link_ids` | `[1000]` | int32 | URDF link id each gripper point belongs to (static) |
| `joint_q` | `[T, 19]` | float32 | robot joint angles (19 joints for ARX5 dual-arm) |
| `openness` | `[T, 2]` | float32 | gripper jaw openness `[left, right]` ∈ [0, 1] |
| `left_ee_pose` | `[T, 7]` | float32 | left EE world pose `(x, y, z, qx, qy, qz, qw)` |
| `right_ee_pose` | `[T, 7]` | float32 | right EE world pose |
| `delta_joint_q` | `[T, 19]` | float32 | per-frame joint angle deltas |
| `delta_openness` | `[T, 2]` | float32 | per-frame openness deltas |
| `delta_left_ee_pos`, `delta_left_ee_quat` | `[T, 3]`, `[T, 4]` | float32 | left EE world-frame deltas |
| `delta_right_ee_pos`, `delta_right_ee_quat` | `[T, 3]`, `[T, 4]` | float32 | right EE world-frame deltas |
| `fps` | scalar | float32 | 60.0 (native sim rate) |
| `record_id` | scalar | str | trajectory identifier |
| `table_top_z` | scalar | float32 | 0.800 (workspace z-height in meters) |

### Heterogeneous N_cloth (important)

Cloth vertex count depends on garment:

| N_cloth | garment | triangulated indices |
|---|---|---|
| 4952 | pants | 28,608 |
| 7021 | short_shirt | 41,301 |
| 10580 | long_shirt | 60,372 |

PointWorld must either:
- (a) handle variable `N` (e.g., pad to a max + valid_length mask), or
- (b) restrict training to one garment (loses coverage)

The recommended approach is (a). ParticleDiffuser pads to `N_TOTAL = max(N_cloth) + 2*1000`.

### T (trajectory length) varies

- bimanual: ~2700-3200 frames per traj
- single_arm: ~1430 per traj
- ik_lift_random / ik_push: ~430 per traj

Pick window starts dynamically per trajectory based on `T`.

---

## 4. Preprocessing (what ParticleDiffuser v8 does)

### Temporal stride = 2

Effective frame rate: 30 Hz (subsample native 60 Hz by 2). When sampling a window, pick frames at indices `[start, start+2, start+4, ...]`.

### Window shape

- `context_len = 2` frames into the past
- `future_len = 16` frames into the future
- Total window = 18 frames at stride 2 = **36 native frames**

### Initial skip

`SKIP_FRAMES = 30` — skip the first 30 native frames of each trajectory (cloth settling time, very low motion).

### Valid window starts

```
start ∈ [SKIP_FRAMES, T - (context_len + future_len) * stride]
      = [30, T - 36]
```

### Importance sampling by cloth motion (recommended)

`diverse_v1` has ~86% of windows with <0.5 mm/frame cloth motion (gripper hasn't reached cloth yet). Uniform sampling wastes most batches on static frames. ParticleDiffuser uses:

```python
# Per-window score = mean per-particle cloth motion (mm) over the window
for window_start in valid_starts:
    mid_frames = cloth_points[window_start : window_start + 36]
    per_frame_motion_mm = ||diff(mid_frames, axis=0)||_2.mean(axis=1) * 1000  # [35]
    score = per_frame_motion_mm.mean()                                         # scalar
    score = max(score, 0.1)                                                    # floor in mm
    weight = score ** 2.0                                                      # alpha=2.0
probs = weights / weights.sum()                                                # multinomial
```

Then sample window starts via `np.random.choice(starts, p=probs)`.

This brings the <0.5 mm static fraction from 86% → ~45%. **Strongly recommended** for the comparison to be apples-to-apples; if PointWorld uses uniform sampling, val_loss numbers will not be comparable to v8's.

### Normalization

ParticleDiffuser uses **min-max to [-1, 1]** with clipping, per-axis:

```python
x_normalized = clip(2 * (x - min) / (max - min) - 1, -1, 1)
```

Stats are computed on the full diverse_v1 training set. PointWorld can either:
- Recompute its own per-axis stats on the train split (cleanest)
- Or use the same scheme but with its own preferred normalization (document the choice)

The choice affects raw loss values but not relative comparison if both teams normalize their own predictions consistently.

---

## 5. Action representation

v8 uses **world-frame pose deltas (20-D)** per pair of grippers:

```
per-frame action = [Δt_left(3), ΔR6d_left(6), Δopen_left(1),
                    Δt_right(3), ΔR6d_right(6), Δopen_right(1)]
```

Where:
- `Δt` = translation delta from previous frame (3-D, meters)
- `ΔR6d` = orientation delta as 6-D representation (first two columns of `R_curr @ R_prev^T`)
- `Δopen` = openness delta

Plus anchor pose at last context frame:
```
anchor = [t(3), R6d(6), open(1)] × 2 grippers = [2, 10]
```

The model integrates deltas from the anchor to recover absolute gripper trajectory.

**PointWorld is free to use a different action encoding** (joint_q, raw poses, etc.) — just document the choice. Comparison is via cloth-prediction quality, not action conditioning fidelity.

---

## 6. Evaluation protocol

### 6a. Val_loss during training

- Log on the 8 val trajs (Section 2)
- Run every 500 train steps
- Report the best val_loss + final val_loss

### 6b. Long-horizon rollout (the head-to-head paper number)

**On a trajectory that's fully OOD for any model trained on diverse_v1:**

```
/data/group_data/katefgroup/VLA/3DPCWM/robert_sim1/sim1_cloth_variant_0002/pointcloud_dataset/000086.npz
```

(This is from a separate, older dataset. Models trained on `diverse_v1` have never seen any traj from `sim1_cloth_variant_0002`.)

**Setup**:
- `start_frame = 30`
- `context_len = 2` (= native frames 30, 32 at stride 2)
- `num_frames = 1000` (output prediction length)
- `temporal_stride = 2`

**Output**: `pred_1000frames.npy` with shape `[1002, 9021, 3]`:
- `1002 = 2 context + 1000 predicted` (context echoed in output)
- `9021 = 7021 cloth + 1000 left_ee + 1000 right_ee`
- Order: cloth points first, then left_ee, then right_ee

This must match exactly so a single comparison script can score both predictions.

### 6c. Metrics computed on the 1000-frame rollout

For comparison, both teams report (computed by a single shared script):
- **Cloth L2 to GT (mm)** at frames {10, 50, 100, 500, 1000}
- **Mesh edge distortion (mm)** at the same frames (mean over K=8 KNN edges, computed on first-frame topology)

**v8 reference numbers** (for context):

| metric | v8 |
|---|---|
| best val_loss (on old 4-traj val set) | 0.082 |
| cloth L2 @ 1000 frames (variant_0002/000086) | 12.42 mm |
| mesh edge distortion @ 1000 frames | ~12 mm |

---

## 7. Reporting back

Three files needed from the PointWorld team:

1. **`pred_1000frames.npy`** with shape `[1002, 9021, 3]` from the rollout in Section 6b
2. **Final val_loss** on the 8-traj val set (single number, ideally with curve)
3. **Training summary** (1 paragraph): action encoding used, normalization choice, importance-sampling on/off, number of training steps, hardware

That's enough for a fair head-to-head table in the paper.

---

## 8. Common pitfalls to check

1. **Stride mismatch**: forgetting `temporal_stride=2` doubles the effective horizon length — predictions look 2× worse.
2. **N_cloth handling**: model that assumes fixed N will silently produce wrong outputs for the wrong-garment trajs.
3. **SKIP_FRAMES**: if you sample from frame 0, you waste training on cloth-settling frames where nothing happens.
4. **Action coordinate frame**: deltas are world-frame, not body-frame. Mixing these silently breaks gripper rigidity.
5. **Importance sampling off**: val_loss will appear ~30% lower than ours because most train batches are static-cloth.
6. **Val traj contamination**: verify the 8-traj list is held out at every dataloader / cache / sampler stage.

---

## 9. Dataset metadata cache (optional speed-up)

The dataset directory has a precomputed cache:

```
/data/group_data/katefgroup/VLA/3DPCWM/robert_sim1/sim1_cloth_diverse_v1/_meta_cache.json
```

It maps each `.npz` path to `{n_cloth, n_ee, T}` so you don't need to open each file to discover its shape. ~600 KB. Useful for filtering by garment without scanning data.

---

## 10. Asset paths (for reference / sanity)

- Robot URDF (ARX5 bimanual): not needed for cloth prediction; only relevant for FK if you want to compute EE poses from joint angles
- Cloth meshes: embedded in each `.npz` via `cloth_points` topology — no separate file needed
- Workspace: table top at `z = 0.800` m, cloth typically rests at `z ≈ 0.81 m`

---

## 11. Optional: gripper template files

**Not required** for a vanilla baseline. The per-frame gripper point clouds are already in each `.npz` as `left_ee_points` / `right_ee_points` (1000 points each, world-frame).

The template files are useful only if PointWorld wants to:

1. **Rigid-snap the predicted gripper points** (post-process the 1000 free points by projecting them onto the canonical template per link via Kabsch — keeps gripper exactly rigid). ParticleDiffuser v8 does this.
2. **Predict SE(3) per gripper instead of 1000 free points** — needs the template to FK back to 1000 points for evaluation/visualization.
3. **Compute per-link metrics** — `template_link_ids` says which of the 1000 points belongs to which URDF link.

### Files

```
/data/group_data/katefgroup/VLA/3DPCWM/robert_sim1/sim1_cloth_variant_0002/gripper_template.npz          # 17KB
/data/group_data/katefgroup/VLA/3DPCWM/robert_sim1/sim1_cloth_variant_0002/gripper_template_aligned.npz  # 17KB
```

### Schema

```python
# gripper_template.npz
template_points    : [1000, 3]  float32   # canonical gripper point cloud (URDF-sampled)
template_link_ids  : [1000]     int32     # URDF link index per point (3 unique links per side)
link_names_left    : [3]        str       # ARX5 left-arm link names
link_names_right   : [3]        str       # ARX5 right-arm link names

# gripper_template_aligned.npz — same fields plus:
reference_openness : scalar     float32   # openness state the template was rendered at
```

Both files are included in the handoff zip for convenience.
