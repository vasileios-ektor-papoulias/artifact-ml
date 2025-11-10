from abc import abstractmethod
from typing import Any, Generic, Optional, Type, TypeVar

from artifact_core.libs.resource_spec.binary.protocol import BinaryFeatureSpecProtocol

from artifact_torch.base.components.routines.loader import DataLoaderRoutine
from artifact_torch.base.components.routines.train_diagnostics import TrainDiagnosticsRoutine
from artifact_torch.base.experiment.experiment import Experiment
from artifact_torch.base.model.io import ModelInput, ModelOutput
from artifact_torch.base.trainer.trainer import Trainer
from artifact_torch.binary_classification.model import BinaryClassifier
from artifact_torch.binary_classification.routine import (
    BinaryClassificationRoutine,
    BinaryClassificationRoutineData,
)
from artifact_torch.core.model.classifier import ClassificationParams

BinaryClassifierTContr = TypeVar(
    "BinaryClassifierTContr", bound=BinaryClassifier[Any, Any, Any, Any]
)
ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput)
ClassificationParamsTContr = TypeVar("ClassificationParamsTContr", bound=ClassificationParams)
ClassificationDataTContr = TypeVar("ClassificationDataTContr")
BinaryClassificationExperimentT = TypeVar(
    "BinaryClassificationExperimentT", bound="BinaryClassificationExperiment"
)


class BinaryClassificationExperiment(
    Experiment[
        BinaryClassifierTContr,
        ModelInputTContr,
        ModelOutputTContr,
        BinaryClassificationRoutineData[ClassificationDataTContr],
        BinaryFeatureSpecProtocol,
    ],
    Generic[
        BinaryClassifierTContr,
        ModelInputTContr,
        ModelOutputTContr,
        ClassificationParamsTContr,
        ClassificationDataTContr,
    ],
):
    @classmethod
    @abstractmethod
    def _get_trainer(
        cls,
    ) -> Type[
        Trainer[
            BinaryClassifierTContr,
            ModelInputTContr,
            ModelOutputTContr,
            Any,
            Any,
        ]
    ]: ...

    @classmethod
    @abstractmethod
    def _get_train_diagnostics_routine(
        cls,
    ) -> Optional[
        Type[TrainDiagnosticsRoutine[BinaryClassifierTContr, ModelInputTContr, ModelOutputTContr]]
    ]: ...

    @classmethod
    @abstractmethod
    def _get_loader_routine(
        cls,
    ) -> Optional[
        Type[DataLoaderRoutine[BinaryClassifierTContr, ModelInputTContr, ModelOutputTContr]]
    ]: ...

    @classmethod
    @abstractmethod
    def _get_artifact_routine(
        cls,
    ) -> Optional[
        Type[BinaryClassificationRoutine[ClassificationParamsTContr, ClassificationDataTContr]]
    ]: ...
