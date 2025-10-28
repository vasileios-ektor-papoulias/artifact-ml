from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Mapping, Optional, TypeVar

import pandas as pd
from artifact_core.binary_classification.artifacts.base import BinaryClassificationArtifactResources
from artifact_core.libs.resource_spec.binary.protocol import BinaryFeatureSpecProtocol
from artifact_core.libs.resources.categorical.category_store.binary import BinaryCategoryStore
from artifact_experiment.base.data_split import DataSplit, DataSplitSuffixAppender
from artifact_experiment.base.tracking.client import TrackingClient
from artifact_experiment.binary_classification.validation_plan import BinaryClassificationPlan

from artifact_torch.base.components.routines.artifact import (
    ArtifactRoutine,
    ArtifactRoutineData,
    ArtifactRoutineHyperparams,
)
from artifact_torch.binary_classification.model import BinaryClassifier
from artifact_torch.core.model.classifier import ClassificationParams
from artifact_torch.libs.exports.metadata import MetadataExporter

ClassificationParamsTCov = TypeVar("ClassificationParamsTCov", bound=ClassificationParams)
BinaryClassificationRoutineT = TypeVar(
    "BinaryClassificationRoutineT", bound="BinaryClassificationRoutine"
)


@dataclass
class BinaryClassificationRoutineHyperparams(
    ArtifactRoutineHyperparams, Generic[ClassificationParamsTCov]
):
    classification_params: ClassificationParamsTCov


@dataclass
class BinaryClassificationRoutineData(ArtifactRoutineData):
    true_category_store: BinaryCategoryStore
    classification_data: pd.DataFrame


class BinaryClassificationRoutine(
    ArtifactRoutine[
        BinaryClassifier[Any, Any, ClassificationParamsTCov],
        BinaryClassificationRoutineHyperparams[ClassificationParamsTCov],
        BinaryClassificationRoutineData,
        BinaryClassificationArtifactResources,
        BinaryFeatureSpecProtocol,
    ],
    Generic[ClassificationParamsTCov],
):
    _resource_export_prefix = "CLASSIFICATION_RESULTS"

    @classmethod
    @abstractmethod
    def _get_periods(cls) -> Mapping[DataSplit, int]: ...

    @classmethod
    @abstractmethod
    def _get_validation_plans(
        cls,
        artifact_resource_spec: BinaryFeatureSpecProtocol,
        tracking_client: Optional[TrackingClient],
    ) -> Mapping[DataSplit, BinaryClassificationPlan]: ...

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
        self, model: BinaryClassifier[Any, Any, ClassificationParamsTCov]
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
        n_epochs_elapsed: int,
        data_split: DataSplit,
        tracking_client: TrackingClient,
    ):
        true = artifact_resources.true_category_store.id_to_category
        true = {str(identifier): category for identifier, category in true.items()}
        predicted = artifact_resources.classification_results.id_to_predicted_category
        predicted = {str(identifier): category for identifier, category in predicted.items()}
        probs = artifact_resources.classification_results.id_to_prob_pos
        probs = {str(identifier): prob for identifier, prob in probs.items()}
        dict_resources = {
            identifier: {
                "true": true.get(identifier),
                "predicted": predicted.get(identifier),
                "prob_pos": probs.get(identifier),
            }
            for identifier in sorted(set(true) | set(predicted) | set(probs), key=int)
        }
        resource_export_prefix = cls._get_resource_export_prefix(data_split=data_split)
        MetadataExporter.export(
            data=dict_resources,
            tracking_client=tracking_client,
            prefix=resource_export_prefix,
            step=n_epochs_elapsed,
        )

    @classmethod
    def _get_resource_export_prefix(cls, data_split: DataSplit) -> str:
        return DataSplitSuffixAppender.append_suffix(
            name=cls._resource_export_prefix, data_split=data_split
        )
