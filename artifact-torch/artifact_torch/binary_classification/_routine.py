from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Mapping, Optional, Type, TypeVar

from artifact_core._libs.resources_spec.binary.protocol import BinaryFeatureSpecProtocol
from artifact_core._libs.resourcess.categorical.category_store.binary import BinaryCategoryStore
from artifact_core.binary_classification._artifacts.base import (
    BinaryClassificationArtifactResources,
)
from artifact_experiment.base.tracking.background.tracking_queue import TrackingQueue
from artifact_experiment.base.types.data_split import DataSplit
from artifact_experiment.binary_classification.plan import BinaryClassificationPlan

from artifact_torch.base.components.callbacks.export import ExportCallback, ExportCallbackResources
from artifact_torch.base.components.routines.artifact import (
    ArtifactRoutine,
    ArtifactRoutineData,
    ArtifactRoutineHyperparams,
)
from artifact_torch.binary_classification._model import BinaryClassifier
from artifact_torch.core.model.classifier import ClassificationParams
from artifact_torch.libs.components.callbacks.export.classification_results import (
    ClassificationResultsExportCallback,
)

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
    true_category_store: BinaryCategoryStore
    classification_data: ClassificationDataTCov


ClassificationDataTContr = TypeVar("ClassificationDataTContr", contravariant=True)
BinaryClassificationRoutineT = TypeVar(
    "BinaryClassificationRoutineT", bound="BinaryClassificationRoutine"
)


class BinaryClassificationRoutine(
    ArtifactRoutine[
        BinaryClassifier[Any, Any, ClassificationParamsTCov, Any],
        BinaryClassificationRoutineHyperparams[ClassificationParamsTCov],
        BinaryClassificationRoutineData[ClassificationDataTContr],
        BinaryClassificationArtifactResources,
        BinaryFeatureSpecProtocol,
        Dict[str, Any],
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
    def _get_classification_params(cls) -> ClassificationParamsTCov: ...

    @classmethod
    @abstractmethod
    def _get_artifact_plan(
        cls, data_split: DataSplit
    ) -> Optional[Type[BinaryClassificationPlan]]: ...

    @classmethod
    def _get_hyperparams(cls) -> BinaryClassificationRoutineHyperparams[ClassificationParamsTCov]:
        classification_params = cls._get_classification_params()
        hyperparams = BinaryClassificationRoutineHyperparams[ClassificationParamsTCov](
            classification_params=classification_params
        )
        return hyperparams

    @classmethod
    def _get_export_callback(
        cls, tracking_queue: Optional[TrackingQueue]
    ) -> Optional[ExportCallback[Dict[str, Any]]]:
        if tracking_queue is not None:
            return ClassificationResultsExportCallback(period=1, writer=tracking_queue.file_writer)

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
                true_category_store=self._data[data_split].true_category_store,
                classification_results=classification_results,
            )
            resources_by_split[data_split] = resources
        return resources_by_split

    @classmethod
    def _export_artifact_resources(
        cls,
        artifact_resources: BinaryClassificationArtifactResources,
        export_callback: ExportCallback[Dict[str, Any]],
        n_epochs_elapsed: int,
        data_split: DataSplit,
    ):
        true = artifact_resources.true_category_store.id_to_category
        true = {str(identifier): category for identifier, category in true.items()}
        predicted = artifact_resources.classification_results.id_to_predicted_category
        predicted = {str(identifier): category for identifier, category in predicted.items()}
        probs = artifact_resources.classification_results.id_to_prob_pos
        probs = {str(identifier): prob for identifier, prob in probs.items()}
        dict_artifact_resources = {
            identifier: {
                "true": true.get(identifier),
                "predicted": predicted.get(identifier),
                "prob_pos": probs.get(identifier),
            }
            for identifier in sorted(set(true) | set(predicted) | set(probs), key=int)
        }
        export_callback_resources = ExportCallbackResources[Dict[str, Any]](
            step=n_epochs_elapsed, export_data=dict_artifact_resources, data_split=data_split
        )
        export_callback.execute(resources=export_callback_resources)
