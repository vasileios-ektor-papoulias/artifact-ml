from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from artifact_experiment.base.tracking.client import TrackingClient

from artifact_torch.base.components.callbacks.batch import (
    BatchCallback,
    BatchCallbackHandler,
    BatchCallbackResources,
)
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
        ls_callbacks = cls._get_batch_callbacks(tracking_client=tracking_client)
        routine = cls._build(ls_callbacks=ls_callbacks)
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
        self, model_input: ModelInputTContr, model_output: ModelOutputTContr, batch_idx: int
    ):
        resources = BatchCallbackResources[ModelInputTContr, ModelOutputTContr](
            step=batch_idx, model_input=model_input, model_output=model_output
        )
        self._handler.execute(resources=resources)

    @classmethod
    def _build(
        cls: Type[BatchRoutineT],
        ls_callbacks: List[BatchCallback[ModelInputTContr, ModelOutputTContr, Any]],
    ) -> BatchRoutineT:
        handler = BatchCallbackHandler[ModelInputTContr, ModelOutputTContr, Any](
            ls_callbacks=ls_callbacks,
        )
        subroutine = cls(
            handler=handler,
        )
        return subroutine
