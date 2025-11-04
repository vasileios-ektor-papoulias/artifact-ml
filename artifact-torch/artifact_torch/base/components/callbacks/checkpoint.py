from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

from artifact_experiment.base.tracking.client import TrackingClient

from artifact_torch.base.components.callbacks.periodic import (
    PeriodicCallback,
    PeriodicCallbackResources,
)


@dataclass(frozen=True)
class CheckpointCallbackResources(PeriodicCallbackResources):
    checkpoint: Dict[str, Any]


class CheckpointCallback(PeriodicCallback[CheckpointCallbackResources]):
    def __init__(self, period: int, tracking_client: TrackingClient):
        super().__init__(period=period)
        self._tracking_client = tracking_client

    @abstractmethod
    def _export(self, checkpoint: Dict[str, Any], step: int): ...

    def _execute(self, resources: CheckpointCallbackResources):
        if self._tracking_client is not None:
            self._export(
                checkpoint=resources.checkpoint,
                step=resources.step,
            )
