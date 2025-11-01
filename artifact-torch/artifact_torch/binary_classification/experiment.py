from abc import abstractmethod
from typing import Any, Generic, Mapping, Optional, Type, TypeVar

from artifact_core.libs.resource_spec.binary.protocol import BinaryFeatureSpecProtocol
from artifact_experiment.base.data_split import DataSplit
from artifact_experiment.base.tracking.client import TrackingClient

from artifact_torch.base.components.routines.batch import BatchRoutine
from artifact_torch.base.components.routines.loader import DataLoaderRoutine
from artifact_torch.base.components.routines.loader_hook import DataLoaderHookRoutine
from artifact_torch.base.experiment.experiment import Experiment
from artifact_torch.base.model.io import ModelInput, ModelOutput
from artifact_torch.base.trainer.trainer import Trainer
from artifact_torch.binary_classification.model import BinaryClassifier
from artifact_torch.binary_classification.routine import (
    BinaryClassificationRoutine,
    BinaryClassificationRoutineData,
)
from artifact_torch.core.model.classifier import ClassificationParams

BinaryClassifierT = TypeVar("BinaryClassifierT", bound=BinaryClassifier[Any, Any, Any, Any])
ModelInputT = TypeVar("ModelInputT", bound=ModelInput)
ModelOutputT = TypeVar("ModelOutputT", bound=ModelOutput)
ClassificationParamsT = TypeVar("ClassificationParamsT", bound=ClassificationParams)
ClassificationDataT = TypeVar("ClassificationDataT")
BinaryClassificationExperimentT = TypeVar(
    "BinaryClassificationExperimentT", bound="BinaryClassificationExperiment"
)


class BinaryClassificationExperiment(
    Experiment[
        BinaryClassifierT,
        ModelInputT,
        ModelOutputT,
        BinaryClassificationRoutineData[ClassificationDataT],
        BinaryFeatureSpecProtocol,
    ],
    Generic[
        BinaryClassifierT, ModelInputT, ModelOutputT, ClassificationParamsT, ClassificationDataT
    ],
):
    @classmethod
    @abstractmethod
    def _get_trainer_type(
        cls,
    ) -> Type[
        Trainer[
            BinaryClassifierT,
            ModelInputT,
            ModelOutputT,
            Any,
            Any,
        ]
    ]: ...

    @classmethod
    @abstractmethod
    def _get_batch_routine(
        cls,
        tracking_client: Optional[TrackingClient] = None,
    ) -> Optional[BatchRoutine[ModelInputT, ModelOutputT]]: ...

    @classmethod
    @abstractmethod
    def _get_loader_routine(
        cls,
        data_split: DataSplit,
        tracking_client: Optional[TrackingClient] = None,
    ) -> Optional[DataLoaderRoutine[ModelInputT, ModelOutputT]]: ...

    @classmethod
    @abstractmethod
    def _get_loader_hook_routine(
        cls,
        data_split: DataSplit,
        tracking_client: Optional[TrackingClient] = None,
    ) -> Optional[DataLoaderHookRoutine[BinaryClassifierT, ModelInputT, ModelOutputT]]: ...

    @classmethod
    @abstractmethod
    def _get_artifact_routine(
        cls,
        data: Mapping[DataSplit, BinaryClassificationRoutineData[ClassificationDataT]],
        data_spec: BinaryFeatureSpecProtocol,
        tracking_client: Optional[TrackingClient] = None,
    ) -> Optional[BinaryClassificationRoutine[ClassificationParamsT, ClassificationDataT]]: ...
