# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import torch.distributed as dist
from utils import _print
from pointworld.checkpoint_contract import attach_checkpoint_contract


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def save_checkpoint_now(trainer, adjusted_batch_count, log_dict=None):
    if trainer.args.distributed:
        dist.barrier()
    if trainer.rank == 0:
        _print("Saving checkpoint")
        save_checkpoint(trainer, log_dict or {})
    if trainer.args.distributed:
        dist.barrier()


def save_checkpoint(trainer, log_dict=None):
    if trainer.rank != 0:
        return

    save_dict = {
        "model": _unwrap_model(trainer.model).state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "args": trainer.args,
        "exp_name": trainer.exp_name,
        "wandb_id": trainer.wandb_id,
        "batch_count": trainer.batch_count,
        "epoch_count": trainer.epoch_count,
        "sample_count": trainer.sample_count,
        "optimizer_step_count": getattr(trainer, "optimizer_step_count", 0),
    }
    attach_checkpoint_contract(save_dict, args=trainer.args, context="training save checkpoint")

    # latest.pt is the rolling checkpoint a resubmitted leg resumes from. It is
    # written to a temporary file in the same directory and renamed, so a job
    # that dies mid-write leaves the previous checkpoint intact rather than a
    # truncated one. The mode is set explicitly: a 0600 checkpoint is unreadable
    # to the rest of the group.
    final_local_path = os.path.join(trainer.save_dir, "latest.pt")
    tmp_path = final_local_path + ".tmp"
    torch.save(save_dict, tmp_path)
    os.chmod(tmp_path, 0o640)
    os.replace(tmp_path, final_local_path)

    # The repo's other scripts (eval, the older sbatches) look for model-last.pt.
    legacy = os.path.join(trainer.save_dir, "model-last.pt")
    if not os.path.islink(legacy):
        if os.path.exists(legacy):
            os.remove(legacy)
        os.symlink("latest.pt", legacy)

    return trainer.save_dir


def save_matched_snapshot(trainer, matched_step):
    """Copy the rolling checkpoint to a permanent, budget-named snapshot.

    `matched_step` counts steps of the run this baseline is matched against
    (args.matched_windows_per_step windows each), so the snapshots land at the
    same budget whatever batch size or number of GPUs a leg happens to use.
    Copied rather than serialized again: the state is already on disk.
    """
    if trainer.rank != 0:
        return None
    import shutil
    latest = os.path.join(trainer.save_dir, "latest.pt")
    if not os.path.exists(latest):
        return None
    snapshot = os.path.join(trainer.save_dir, f"matched_step_{int(matched_step):06d}.pt")
    tmp = snapshot + ".tmp"
    shutil.copyfile(latest, tmp)
    os.chmod(tmp, 0o640)
    os.replace(tmp, snapshot)
    _print(f"Snapshot at matched step {int(matched_step):,}: {snapshot}")
    return snapshot


def load_checkpoint_from_path(trainer, model_path=None):
    if model_path is None:
        if trainer.rank == 0:
            _print("No checkpoint path provided, starting from scratch")
        return None
    assert os.path.exists(model_path), f"Checkpoint not found: {model_path}"
    if trainer.rank == 0:
        _print(f"Loading checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if "wandb_id" in checkpoint:
        checkpoint.pop("wandb_id")
    if "exp_name" in checkpoint:
        checkpoint.pop("exp_name")
    return checkpoint


def load_checkpoint(trainer, checkpoint):
    model_state_dict = checkpoint["model"]
    expected_keys = set(_unwrap_model(trainer.model).state_dict().keys())
    ckpt_keys = set(model_state_dict.keys())
    if ckpt_keys != expected_keys:
        missing = sorted(expected_keys - ckpt_keys)
        extra = sorted(ckpt_keys - expected_keys)
        missing_suffix = "..." if len(missing) > 10 else ""
        extra_suffix = "..." if len(extra) > 10 else ""
        raise RuntimeError(
            "Checkpoint model keys do not match the current model state_dict. "
            "Ensure the distributed/Non-DDP setting and model config match. "
            f"Missing keys: {missing[:10]}{missing_suffix}. "
            f"Extra keys: {extra[:10]}{extra_suffix}."
        )
    _unwrap_model(trainer.model).load_state_dict(model_state_dict)

    if not trainer.inference_only:
        if "optimizer" not in checkpoint:
            if getattr(trainer.args, "allow_optimizer_reset", False):
                _print("Checkpoint missing optimizer state; resetting optimizer as requested.")
            else:
                raise RuntimeError(
                    "Checkpoint missing optimizer state. "
                    "Rerun with --allow_optimizer_reset=true to skip optimizer restore."
                )
        else:
            try:
                trainer.optimizer.load_state_dict(checkpoint["optimizer"])
            except ValueError as exc:
                if getattr(trainer.args, "allow_optimizer_reset", False):
                    _print(
                        "Optimizer state incompatible with current optimizer; "
                        "resetting optimizer state as requested."
                    )
                else:
                    raise RuntimeError(
                        "Optimizer state incompatible with current optimizer. "
                        "Rerun with --allow_optimizer_reset=true to skip optimizer restore."
                    ) from exc
    assert "batch_count" in checkpoint, "Checkpoint missing batch_count."
    assert "epoch_count" in checkpoint, "Checkpoint missing epoch_count."
    assert "sample_count" in checkpoint, "Checkpoint missing sample_count."
    trainer.batch_count = checkpoint["batch_count"]
    trainer.epoch_count = checkpoint["epoch_count"]
    trainer.sample_count = checkpoint["sample_count"]
    # absent in checkpoints written before gradient accumulation existed
    trainer.optimizer_step_count = int(checkpoint.get("optimizer_step_count", checkpoint["batch_count"]))
