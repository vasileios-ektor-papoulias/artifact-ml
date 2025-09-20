from dataclasses import dataclass
from typing import Dict, Generic, List, Optional, Type, TypeVar

from artifact_core.binary_classification.artifacts.base import ClassificationArtifactResources
from artifact_core.libs.resource_spec.categorical.protocol import CategoricalFeatureSpecProtocol
from artifact_core.libs.resources.base.resource_store import IdentifierType
from artifact_core.libs.resources.classification.classification_results import ClassificationResults
from numpy import ndarray

from artifact_experiment.base.callbacks.artifact import ArtifactCallbackResources

CategoricalFeatureSpecProtocolT = TypeVar(
    "CategoricalFeatureSpecProtocolT", bound=CategoricalFeatureSpecProtocol
)
ClassificationCallbackResourcesT = TypeVar(
    "ClassificationCallbackResourcesT", bound="ClassificationCallbackResources"
)


@dataclass
class ClassificationCallbackResources(
    ArtifactCallbackResources[ClassificationArtifactResources[CategoricalFeatureSpecProtocolT]],
    Generic[CategoricalFeatureSpecProtocolT],
):
    @classmethod
    def build(
        cls: Type[ClassificationCallbackResourcesT],
        ls_categories: List[str],
        id_to_category: Dict[IdentifierType, str],
        id_to_logits: Optional[Dict[IdentifierType, ndarray]] = None,
    ) -> ClassificationCallbackResourcesT:
        artifact_resources = ClassificationArtifactResources.build(
            ls_categories=ls_categories, id_to_category=id_to_category, id_to_logits=id_to_logits
        )
        callback_resources = cls(artifact_resources=artifact_resources)
        return callback_resources

    @classmethod
    def from_classification_results(
        cls: Type[ClassificationCallbackResourcesT],
        classification_results: ClassificationResults[CategoricalFeatureSpecProtocolT],
    ) -> ClassificationCallbackResourcesT:
        artifact_resources = ClassificationArtifactResources(
            classification_results=classification_results
        )
        callback_resources = cls(artifact_resources=artifact_resources)
        return callback_resources
