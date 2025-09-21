from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Optional, Type, TypeVar

import pandas as pd
from artifact_core.binary_classification.artifacts.base import BinaryClassificationArtifactResources
from artifact_core.libs.resource_spec.binary.protocol import BinaryFeatureSpecProtocol
from artifact_core.libs.resources.categorical.category_store.binary import BinaryCategoryStore
from artifact_experiment.base.tracking.client import TrackingClient
from artifact_experiment.binary_classification.validation_plan import BinaryClassifierEvaluationPlan

from artifact_torch.base.components.routines.artifact import (
    ArtifactRoutineData,
    ArtifactRoutineHyperparams,
    ArtifactValidationRoutine,
)
from artifact_torch.binary_classification.model import BinaryClassifier
from artifact_torch.core.model.classifier import ClassificationParams
from artifact_torch.libs.exports.metadata import MetadataExporter

ClassificationParamsT = TypeVar("ClassificationParamsT", bound=ClassificationParams)
BinaryClassificationRoutineT = TypeVar(
    "BinaryClassificationRoutineT", bound="BinaryClassificationRoutine"
)


@dataclass
class BinaryClassificationRoutineHyperparams(
    ArtifactRoutineHyperparams, Generic[ClassificationParamsT]
):
    classification_params: ClassificationParamsT


@dataclass
class BinaryClassificationRoutineData(ArtifactRoutineData):
    true_category_store: BinaryCategoryStore
    classification_data: pd.DataFrame


class BinaryClassificationRoutine(
    ArtifactValidationRoutine[
        BinaryClassifier[Any, Any, ClassificationParamsT],
        BinaryClassificationRoutineHyperparams[ClassificationParamsT],
        BinaryClassificationRoutineData,
        BinaryClassificationArtifactResources,
        BinaryFeatureSpecProtocol,
    ],
    Generic[ClassificationParamsT],
):
    _resource_export_prefix = "synthetic"

    @classmethod
    def build(
        cls: Type[BinaryClassificationRoutineT],
        true_category_store: BinaryCategoryStore,
        classification_data: pd.DataFrame,
        class_spec: BinaryFeatureSpecProtocol,
        tracking_client: Optional[TrackingClient] = None,
    ) -> BinaryClassificationRoutineT:
        data = BinaryClassificationRoutineData(
            true_category_store=true_category_store, classification_data=classification_data
        )
        routine = cls._build(
            data=data,
            artifact_resource_spec=class_spec,
            tracking_client=tracking_client,
        )
        return routine

    @classmethod
    @abstractmethod
    def _get_period(cls) -> int: ...

    @classmethod
    @abstractmethod
    def _get_classification_params(cls) -> ClassificationParamsT: ...

    @classmethod
    @abstractmethod
    def _get_validation_plan(
        cls,
        artifact_resource_spec: BinaryFeatureSpecProtocol,
        tracking_client: Optional[TrackingClient],
    ) -> BinaryClassifierEvaluationPlan: ...

    @classmethod
    def _get_hyperparams(cls) -> BinaryClassificationRoutineHyperparams[ClassificationParamsT]:
        classification_params = cls._get_classification_params()
        hyperparams = BinaryClassificationRoutineHyperparams[ClassificationParamsT](
            classification_params=classification_params
        )
        return hyperparams

    @classmethod
    def _generate_artifact_resources(
        cls,
        model: BinaryClassifier[Any, Any, ClassificationParamsT],
        hyperparams: BinaryClassificationRoutineHyperparams,
        data: BinaryClassificationRoutineData,
    ) -> BinaryClassificationArtifactResources:
        classification_results = model.classify(
            data=data.classification_data, params=hyperparams.classification_params
        )
        resources = BinaryClassificationArtifactResources.from_stores(
            true_category_store=data.true_category_store,
            classification_results=classification_results,
        )
        return resources

    @classmethod
    def _export_artifact_resources(
        cls,
        artifact_resources: BinaryClassificationArtifactResources,
        n_epochs_elapsed: int,
        tracking_client: TrackingClient,
    ):
        true = artifact_resources.true_category_store.id_to_category
        true = {str(identifier): category for identifier, category in true.items()}
        predicted = artifact_resources.classification_results.id_to_predicted_category
        predicted = {str(identifier): category for identifier, category in predicted.items()}
        probs = artifact_resources.classification_results.id_to_probs
        probs = {str(identifier): arr_probs.tolist() for identifier, arr_probs in probs.items()}
        dict_resources = {
            identifier: {
                "true": true.get(identifier),
                "predicted": predicted.get(identifier),
                "probs": probs.get(identifier),
            }
            for identifier in set(true) | set(predicted) | set(probs)
        }
        MetadataExporter.export(
            data=dict_resources,
            tracking_client=tracking_client,
            prefix=cls._resource_export_prefix,
            step=n_epochs_elapsed,
        )
