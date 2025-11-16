from dataclasses import dataclass
from typing import Any, Mapping

from artifact_experiment.spi.resources import ExportCallbackResources


@dataclass(frozen=True)
class CheckpointCallbackResources(ExportCallbackResources[Mapping[str, Any]]):
    epoch: int
