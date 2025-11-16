from typing import List, Mapping, Optional, Type, TypeVar

from artifact_core.binary_classification.collections import (
    BinaryClassificationResults,
    BinaryClassStore,
)
from artifact_core.binary_classification.spi import (
    BinaryClassificationArtifactResources,
    BinaryClassSpecProtocol,
)
from artifact_core.typing import IdentifierType

from artifact_experiment._domains.classification.callback_resources import (
    ClassificationCallbackResources,
)

ClassificationCallbackResourcesT = TypeVar(
    "ClassificationCallbackResourcesT", bound=ClassificationCallbackResources
)


class BinaryClassificationCallbackResources(
    ClassificationCallbackResources[BinaryClassificationArtifactResources]
):
    @classmethod
    def build(
        cls: Type[ClassificationCallbackResourcesT],
        class_names: List[str],
        positive_class: str,
        true: Mapping[IdentifierType, str],
        predicted: Mapping[IdentifierType, str],
        probs_pos: Optional[Mapping[IdentifierType, float]] = None,
    ) -> ClassificationCallbackResourcesT:
        artifact_resources = BinaryClassificationArtifactResources.build(
            class_names=class_names,
            positive_class=positive_class,
            true=true,
            predicted=predicted,
            probs_pos=probs_pos,
        )
        callback_resources = cls(artifact_resources=artifact_resources)
        return callback_resources

    @classmethod
    def from_spec(
        cls: Type[ClassificationCallbackResourcesT],
        class_spec: BinaryClassSpecProtocol,
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
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> ClassificationCallbackResourcesT:
        artifact_resources = BinaryClassificationArtifactResources(
            true_class_store=true_class_store, classification_results=classification_results
        )
        callback_resources = cls(artifact_resources=artifact_resources)
        return callback_resources
