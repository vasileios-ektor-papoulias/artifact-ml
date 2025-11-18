from typing import Any, Mapping, Optional, Sequence, Type, TypeVar

from artifact_core._domains.classification.resources import ClassificationArtifactResources
from artifact_core._libs.resource_specs.binary_classification.protocol import (
    BinaryClassSpecProtocol,
)
from artifact_core._libs.resource_specs.binary_classification.spec import BinaryClassSpec
from artifact_core._libs.resources.binary_classification.class_store import BinaryClassStore
from artifact_core._libs.resources.binary_classification.classification_results import (
    BinaryClassificationResults,
)
from artifact_core._utils.collections.entity_store import IdentifierType

BinaryClassificationArtifactResourcesT = TypeVar(
    "BinaryClassificationArtifactResourcesT", bound="BinaryClassificationArtifactResources"
)


class BinaryClassificationArtifactResources(
    ClassificationArtifactResources[BinaryClassStore, BinaryClassificationResults]
):
    @classmethod
    def build(
        cls: Type[BinaryClassificationArtifactResourcesT],
        class_names: Sequence[str],
        positive_class: str,
        true: Mapping[IdentifierType, str],
        predicted: Mapping[IdentifierType, str],
        probs_pos: Optional[Mapping[IdentifierType, float]] = None,
    ) -> BinaryClassificationArtifactResourcesT:
        class_spec = BinaryClassSpec(class_names=class_names, positive_class=positive_class)
        resources = cls.from_spec(
            class_spec=class_spec, true=true, predicted=predicted, probs_pos=probs_pos
        )
        return resources

    @classmethod
    def from_spec(
        cls: Type[BinaryClassificationArtifactResourcesT],
        class_spec: BinaryClassSpecProtocol,
        true: Mapping[IdentifierType, str],
        predicted: Mapping[IdentifierType, str],
        probs_pos: Optional[Mapping[IdentifierType, float]] = None,
    ) -> BinaryClassificationArtifactResourcesT:
        true_class_store = BinaryClassStore.from_class_names_and_spec(
            class_spec=class_spec, id_to_class=true
        )
        classification_results = BinaryClassificationResults.from_spec(
            class_spec=class_spec, id_to_class=predicted, id_to_prob_pos=probs_pos
        )
        resources = cls(
            true_class_store=true_class_store, classification_results=classification_results
        )
        return resources

    @classmethod
    def from_stores(
        cls: Type[BinaryClassificationArtifactResourcesT],
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> BinaryClassificationArtifactResourcesT:
        artifact_resources = cls(
            true_class_store=true_class_store, classification_results=classification_results
        )
        return artifact_resources

    def serialize(self) -> Mapping[str, Any]:
        true = self.true_class_store.id_to_class_name
        true = {str(identifier): category for identifier, category in true.items()}
        predicted = self.classification_results.id_to_predicted_class
        predicted = {str(identifier): category for identifier, category in predicted.items()}
        probs = self.classification_results.id_to_prob_pos
        probs = {str(identifier): prob for identifier, prob in probs.items()}
        dict_artifact_resources = {
            identifier: {
                "true": true.get(identifier),
                "predicted": predicted.get(identifier),
                "prob_pos": probs.get(identifier),
            }
            for identifier in set(true) | set(predicted) | set(probs)
        }
        return dict_artifact_resources
