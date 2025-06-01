from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from artifact_experiment.base.tracking.client import TrackingClient

from artifact_torch.base.components.callbacks.periodic import (
    PeriodicCallbackResources,
    PeriodicTrackingCallback,
)


@dataclass
class CheckpointCallbackResources(PeriodicCallbackResources):
    checkpoint: Dict[str, Any]


class CheckpointCallback(
    PeriodicTrackingCallback[
        CheckpointCallbackResources,
        Dict[str, Any],
    ],
):
    def __init__(self, period: int, tracking_client: Optional[TrackingClient] = None):
        key = self._get_key()
        super().__init__(key=key, period=period, tracking_client=tracking_client)

    @staticmethod
    @abstractmethod
    def _export(key: str, value: Dict[str, Any], tracking_client: TrackingClient): ...

    @classmethod
    def _get_key(cls) -> str:
        return "checkpoint"

    def _compute(self, resources: CheckpointCallbackResources) -> Dict[str, Any]:
        return resources.checkpoint
