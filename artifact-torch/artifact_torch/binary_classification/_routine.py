from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Mapping, Optional, Type, TypeVar

from artifact_core.binary_classification.collections import BinaryClassStore
from artifact_core.binary_classification.spi import (
    BinaryClassificationArtifactResources,
    BinaryClassSpecProtocol,
)
from artifact_experiment.binary_classification import BinaryClassificationPlan
from artifact_experiment.tracking import DataSplit

from artifact_torch._base.components.routines.artifact import (
    ArtifactRoutine,
    ArtifactRoutineData,
    ArtifactRoutineHyperparams,
)
from artifact_torch._domains.classification.model import ClassificationParams
from artifact_torch.binary_classification._model import BinaryClassifier

ClassificationParamsTCov = TypeVar(
    "ClassificationParamsTCov", bound=ClassificationParams, covariant=True
)


@dataclass
class BinaryClassificationRoutineHyperparams(
    ArtifactRoutineHyperparams, Generic[ClassificationParamsTCov]
):
    classification_params: ClassificationParamsTCov


ClassificationDataTCov = TypeVar("ClassificationDataTCov", covariant=True)


@dataclass
class BinaryClassificationRoutineData(ArtifactRoutineData, Generic[ClassificationDataTCov]):
    true_class_store: BinaryClassStore
    classification_data: ClassificationDataTCov


ClassificationDataTContr = TypeVar("ClassificationDataTContr", contravariant=True)
BinaryClassificationRoutineT = TypeVar(
    "BinaryClassificationRoutineT", bound="BinaryClassificationRoutine"
)


class BinaryClassificationRoutine(
    ArtifactRoutine[
        BinaryClassifier[Any, Any, ClassificationParamsTCov, Any],
        BinaryClassificationRoutineData[ClassificationDataTContr],
        BinaryClassificationRoutineHyperparams[ClassificationParamsTCov],
        BinaryClassificationArtifactResources,
        BinaryClassSpecProtocol,
    ],
    Generic[ClassificationParamsTCov, ClassificationDataTContr],
):
    @classmethod
    @abstractmethod
    def _get_period(
        cls,
        data_split: DataSplit,
    ) -> Optional[int]: ...

    @classmethod
    @abstractmethod
    def _get_artifact_plan(
        cls, data_split: DataSplit
    ) -> Optional[Type[BinaryClassificationPlan]]: ...

    @classmethod
    @abstractmethod
    def _get_classification_params(cls) -> ClassificationParamsTCov: ...

    @classmethod
    def _get_hyperparams(cls) -> BinaryClassificationRoutineHyperparams[ClassificationParamsTCov]:
        classification_params = cls._get_classification_params()
        hyperparams = BinaryClassificationRoutineHyperparams[ClassificationParamsTCov](
            classification_params=classification_params
        )
        return hyperparams

    def _generate_artifact_resources(
        self, model: BinaryClassifier[Any, Any, ClassificationParamsTCov, Any]
    ) -> Mapping[DataSplit, BinaryClassificationArtifactResources]:
        resources_by_split = {}
        for data_split in self._data.keys():
            classification_results = model.classify(
                data=self._data[data_split].classification_data,
                params=self._hyperparams.classification_params,
            )
            resources = BinaryClassificationArtifactResources.from_stores(
                true_class_store=self._data[data_split].true_class_store,
                classification_results=classification_results,
            )
            resources_by_split[data_split] = resources
        return resources_by_split
