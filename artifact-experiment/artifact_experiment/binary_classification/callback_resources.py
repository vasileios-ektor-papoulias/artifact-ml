from typing import List, Mapping, Optional, Type, TypeVar

from artifact_core.binary_classification.spi import (
    BinaryCategoryStore,
    BinaryClassificationArtifactResources,
    BinaryClassificationResults,
    BinaryFeatureSpecProtocol,
    IdentifierType,
)

from artifact_experiment.core.classification.callback_resources import (
    ClassificationCallbackResources,
)

ClassificationCallbackResourcesT = TypeVar(
    "ClassificationCallbackResourcesT", bound="ClassificationCallbackResources"
)


class BinaryClassificationCallbackResources(
    ClassificationCallbackResources[BinaryClassificationArtifactResources]
):
    @classmethod
    def build(
        cls: Type[ClassificationCallbackResourcesT],
        ls_categories: List[str],
        positive_category: str,
        true: Mapping[IdentifierType, str],
        predicted: Mapping[IdentifierType, str],
        probs_pos: Optional[Mapping[IdentifierType, float]] = None,
    ) -> ClassificationCallbackResourcesT:
        artifact_resources = BinaryClassificationArtifactResources.build(
            ls_categories=ls_categories,
            positive_category=positive_category,
            true=true,
            predicted=predicted,
            probs_pos=probs_pos,
        )
        callback_resources = cls(artifact_resources=artifact_resources)
        return callback_resources

    @classmethod
    def from_spec(
        cls: Type[ClassificationCallbackResourcesT],
        class_spec: BinaryFeatureSpecProtocol,
        true: Mapping[IdentifierType, str],
        predicted: Mapping[IdentifierType, str],
        probs_pos: Optional[Mapping[IdentifierType, float]] = None,
    ) -> ClassificationCallbackResourcesT:
        artifact_resources = BinaryClassificationArtifactResources.from_spec(
            class_spec=class_spec, true=true, predicted=predicted, probs_pos=probs_pos
        )
        callback_resources = cls(artifact_resources=artifact_resources)
        return callback_resources

    @classmethod
    def from_stores(
        cls: Type[ClassificationCallbackResourcesT],
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> ClassificationCallbackResourcesT:
        artifact_resources = BinaryClassificationArtifactResources(
            true_category_store=true_category_store, classification_results=classification_results
        )
        callback_resources = cls(artifact_resources=artifact_resources)
        return callback_resources
