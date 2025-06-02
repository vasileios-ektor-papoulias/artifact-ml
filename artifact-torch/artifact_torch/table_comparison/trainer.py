from typing import TypeVar

from artifact_torch.base.components.early_stopping.stopper import StopperUpdateData
from artifact_torch.base.components.model_tracking.tracker import (
    ModelTrackingCriterion,
)
from artifact_torch.base.model.io import ModelInput, ModelOutput
from artifact_torch.core.trainer.trainer import Trainer
from artifact_torch.table_comparison.model import GenerationParams, TabularGenerativeModel
from artifact_torch.table_comparison.validation_routine import TableComparisonValidationRoutine

ModelInputT = TypeVar("ModelInputT", bound=ModelInput)
ModelOutputT = TypeVar("ModelOutputT", bound=ModelOutput)
ModelTrackingCriterionT = TypeVar("ModelTrackingCriterionT", bound=ModelTrackingCriterion)
StopperUpdateDataT = TypeVar("StopperUpdateDataT", bound=StopperUpdateData)
GenerationParamsT = TypeVar("GenerationParamsT", bound=GenerationParams)


TabularGenerativeModelTrainer = Trainer[
    TabularGenerativeModel[ModelInputT, ModelOutputT, GenerationParamsT],
    ModelInputT,
    ModelOutputT,
    TableComparisonValidationRoutine[ModelInputT, ModelOutputT, GenerationParamsT],
    ModelTrackingCriterionT,
    StopperUpdateDataT,
]
