from typing import Any, Dict, Optional

from artifact_experiment.base.tracking.client import TrackingClient

from artifact_torch.base.components.callbacks.checkpoint import CheckpointCallback
from artifact_torch.libs.exports.torch import TorchCheckpointExporter


class StandardCheckpointCallback(CheckpointCallback):
    _default_prefix = "checkpoint"

    def __init__(self, period: int, tracking_client: TrackingClient, prefix: Optional[str] = None):
        super().__init__(period=period, tracking_client=tracking_client)
        if prefix is None:
            prefix = self._default_prefix
        self._prefix = prefix

    @property
    def prefix(self) -> str:
        return self._prefix

    def _export(self, checkpoint: Dict[str, Any], step: int):
        TorchCheckpointExporter.export(
            data=checkpoint,
            tracking_client=self._tracking_client,
            prefix=self._prefix,
            step=step,
        )
