import os
import tempfile
from typing import Any, Dict

import torch
from artifact_experiment.base.tracking.client import TrackingClient

from artifact_torch.base.components.callbacks.checkpoint import CheckpointCallback


class StandardCheckpointExporter:
    _checkpoints_target_dir = "checkpoints"

    @classmethod
    def export(
        cls,
        checkpoint: Dict[str, Any],
        tracking_client: TrackingClient,
        step: int,
    ):
        dir_target = cls._get_checkpoints_target_dir()
        tmp_filename = cls._get_tmp_filename(step=step)
        with tempfile.TemporaryDirectory() as tmpdir:
            path_source = os.path.join(tmpdir, tmp_filename)
            torch.save(checkpoint, path_source)
            tracking_client.upload(path_source=path_source, dir_target=dir_target)

    @classmethod
    def _get_checkpoints_target_dir(cls) -> str:
        return cls._checkpoints_target_dir

    @classmethod
    def _get_tmp_filename(cls, step: int) -> str:
        return f"{step}.json"


class StandardCheckpointCallback(CheckpointCallback):
    @staticmethod
    def _export(checkpoint: Dict[str, Any], tracking_client: TrackingClient, step: int):
        StandardCheckpointExporter.export(
            checkpoint=checkpoint, tracking_client=tracking_client, step=step
        )
