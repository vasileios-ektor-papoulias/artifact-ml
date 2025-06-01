from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from artifact_experiment.base.callbacks.tracking import TrackingCallbackHandler

from artifact_torch.base.components.callbacks.periodic import (
    PeriodicCacheCallback,
    PeriodicCallbackResources,
)
from artifact_torch.base.model.io import (
    ModelInput,
    ModelOutput,
)

CacheDataT = TypeVar("CacheDataT")
ModelInputT = TypeVar("ModelInputT", bound=ModelInput)
ModelOutputT = TypeVar("ModelOutputT", bound=ModelOutput)


@dataclass
class BatchCallbackResources(PeriodicCallbackResources, Generic[ModelInputT, ModelOutputT]):
    model_input: ModelInputT
    model_output: ModelOutputT


class BatchCallback(
    PeriodicCacheCallback[
        BatchCallbackResources[ModelInputT, ModelOutputT],
        CacheDataT,
    ],
    Generic[ModelInputT, ModelOutputT, CacheDataT],
):
    def __init__(self, execution_interval: int):
        key = self._get_key()
        super().__init__(key=key, period=execution_interval)

    @classmethod
    @abstractmethod
    def _get_key(cls) -> str: ...

    @staticmethod
    @abstractmethod
    def _compute_on_batch(model_input: ModelInputT, model_output: ModelOutputT) -> CacheDataT: ...

    def _compute(self, resources: BatchCallbackResources[ModelInputT, ModelOutputT]) -> CacheDataT:
        result = self._compute_on_batch(
            model_input=resources.model_input, model_output=resources.model_output
        )
        return result


class BatchCallbackHandler(
    TrackingCallbackHandler[
        BatchCallback[ModelInputT, ModelOutputT, CacheDataT],
        BatchCallbackResources[ModelInputT, ModelOutputT],
        CacheDataT,
    ],
    Generic[ModelInputT, ModelOutputT, CacheDataT],
):
    pass
