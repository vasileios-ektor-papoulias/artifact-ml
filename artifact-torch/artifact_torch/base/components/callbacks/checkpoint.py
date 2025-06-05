from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from artifact_experiment.base.tracking.client import TrackingClient

from artifact_torch.base.components.callbacks.periodic import (
    PeriodicCallback,
    PeriodicCallbackResources,
)


@dataclass
class CheckpointCallbackResources(PeriodicCallbackResources):
    checkpoint: Dict[str, Any]


class CheckpointCallback(PeriodicCallback[CheckpointCallbackResources]):
    def __init__(self, period: int, tracking_client: Optional[TrackingClient] = None):
        super().__init__(period=period)
        self._tracking_client = tracking_client

    @staticmethod
    @abstractmethod
    def _export(checkpoint: Dict[str, Any], tracking_client: TrackingClient, step: int): ...

    def _execute(self, resources: CheckpointCallbackResources):
        if self._tracking_client is not None:
            self._export(
                checkpoint=resources.checkpoint,
                tracking_client=self._tracking_client,
                step=resources.step,
            )
