from typing import Any, Dict, Generic, Optional, TypeVar

from artifact_torch.base.components.routines.batch import BatchRoutine
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput, ModelOutput

ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)
ModelTContr = TypeVar("ModelTContr", bound=Model, contravariant=True)


class BatchEndFlow(Generic[ModelInputTContr, ModelOutputTContr]):
    def __init__(
        self,
        batch_routine: Optional[BatchRoutine[ModelInputTContr, ModelOutputTContr]] = None,
    ):
        self._batch_routine = batch_routine

    @property
    def cache(self) -> Dict[str, Any]:
        if self._batch_routine is None:
            return {}
        return self._batch_routine.cache

    def execute(
        self,
        model_input: ModelInputTContr,
        model_output: ModelOutputTContr,
        model: Model[ModelInputTContr, ModelOutputTContr],
        batch_idx: int,
    ):
        if self._batch_routine is not None:
            self._batch_routine.execute(
                model_input=model_input,
                model_output=model_output,
                model=model,
                batch_idx=batch_idx,
            )

    def clear_cache(self):
        if self._batch_routine is not None:
            self._batch_routine.clear_cache()
