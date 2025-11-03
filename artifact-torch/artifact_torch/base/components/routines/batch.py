from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Sequence, Type, TypeVar

from artifact_experiment.base.tracking.client import TrackingClient

from artifact_torch.base.components.callbacks.batch import (
    BatchCallback,
    BatchCallbackHandler,
    BatchCallbackResources,
)
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput, ModelOutput

ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)
BatchRoutineT = TypeVar("BatchRoutineT", bound="BatchRoutine")


class BatchRoutine(ABC, Generic[ModelInputTContr, ModelOutputTContr]):
    def __init__(
        self,
        handler: BatchCallbackHandler[ModelInputTContr, ModelOutputTContr, Any],
    ):
        self._handler = handler

    @classmethod
    def build(
        cls: Type[BatchRoutineT], tracking_client: Optional[TrackingClient] = None
    ) -> BatchRoutineT:
        callbacks = cls._get_batch_callbacks(tracking_client=tracking_client)
        routine = cls._build(callbacks=callbacks)
        return routine

    @property
    def cache(self) -> Dict[str, Any]:
        return self._handler.active_cache

    @staticmethod
    @abstractmethod
    def _get_batch_callbacks(
        tracking_client: Optional[TrackingClient],
    ) -> List[BatchCallback[ModelInputTContr, ModelOutputTContr, Any]]: ...

    def execute(
        self,
        model_input: ModelInputTContr,
        model_output: ModelOutputTContr,
        model: Model[ModelInputTContr, ModelOutputTContr],
        batch_idx: int,
    ):
        resources = BatchCallbackResources[ModelInputTContr, ModelOutputTContr](
            step=batch_idx, model_input=model_input, model_output=model_output, model=model
        )
        self._handler.execute(resources=resources)

    def clear_cache(self):
        self._handler.clear()

    @classmethod
    def _build(
        cls: Type[BatchRoutineT],
        callbacks: Sequence[BatchCallback[ModelInputTContr, ModelOutputTContr, Any]],
    ) -> BatchRoutineT:
        handler = BatchCallbackHandler[ModelInputTContr, ModelOutputTContr, Any](
            callbacks=callbacks,
        )
        routine = cls(handler=handler)
        return routine
