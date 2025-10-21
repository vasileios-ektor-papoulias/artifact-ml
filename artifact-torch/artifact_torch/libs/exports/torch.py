import os
import tempfile
from typing import Any, Dict

import torch
from artifact_experiment.base.tracking.client import TrackingClient

from artifact_torch.libs.exports.exporter import Exporter


class TorchCheckpointExporter(Exporter[Dict[str, Any]]):
    _checkpoints_target_dir = "torch_checkpoints"

    @classmethod
    def _export(
        cls, data: Dict[str, Any], tracking_client: TrackingClient, dir_target: str, filename: str
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_source = os.path.join(tmpdir, filename)
            torch.save(data, path_source)
            tracking_client.upload(path_source=path_source, dir_target=dir_target)

    @classmethod
    def _get_target_dir(cls) -> str:
        return cls._checkpoints_target_dir
