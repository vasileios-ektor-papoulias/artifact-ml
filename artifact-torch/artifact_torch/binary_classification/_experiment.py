from abc import abstractmethod
from typing import Any, Generic, Optional, Type, TypeVar

from artifact_core.binary_classification.spi import BinaryClassSpecProtocol

from artifact_torch._base.components.routines.loader import DataLoaderRoutine
from artifact_torch._base.components.routines.train_diagnostics import TrainDiagnosticsRoutine
from artifact_torch._base.experiment.experiment import Experiment
from artifact_torch._base.model.io import ModelInput, ModelOutput
from artifact_torch._base.trainer.trainer import Trainer
from artifact_torch._domains.classification.model import ClassificationParams
from artifact_torch.binary_classification._model import BinaryClassifier
from artifact_torch.binary_classification._routine import (
    BinaryClassificationRoutine,
    BinaryClassificationRoutineData,
)

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
        BinaryClassSpecProtocol,
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
