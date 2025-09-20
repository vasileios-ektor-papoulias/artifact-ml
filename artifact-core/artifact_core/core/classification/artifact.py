from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Generic, List, Optional, Type, TypeVar

from numpy import ndarray

from artifact_core.base.artifact import Artifact
from artifact_core.base.artifact_dependencies import (
    ArtifactHyperparams,
    ArtifactResources,
    ArtifactResult,
)
from artifact_core.libs.resource_spec.categorical.protocol import CategoricalFeatureSpecProtocol
from artifact_core.libs.resources.categorical.distribution_store import IdentifierType
from artifact_core.libs.resources.classification.classification_results import ClassificationResults

ArtifactHyperparamsT = TypeVar("ArtifactHyperparamsT", bound="ArtifactHyperparams")
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)
CategoricalFeatureSpecProtocolT = TypeVar(
    "CategoricalFeatureSpecProtocolT", bound=CategoricalFeatureSpecProtocol
)
ClassificationArtifactResourcesT = TypeVar(
    "ClassificationArtifactResourcesT", bound="ClassificationArtifactResources"
)


@dataclass(frozen=True)
class ClassificationArtifactResources(ArtifactResources, Generic[CategoricalFeatureSpecProtocolT]):
    classification_results: ClassificationResults[CategoricalFeatureSpecProtocolT]

    @classmethod
    def build(
        cls: Type[ClassificationArtifactResourcesT],
        ls_categories: List[str],
        id_to_category: Dict[IdentifierType, str],
        id_to_logits: Optional[Dict[IdentifierType, ndarray]] = None,
    ) -> ClassificationArtifactResourcesT:
        classification_results = ClassificationResults[CategoricalFeatureSpecProtocolT].build(
            ls_categories=ls_categories,
            id_to_category=id_to_category,
            id_to_logits=id_to_logits,
        )
        resources = cls(classification_results=classification_results)
        return resources


class ClassificationArtifact(
    Artifact[
        ClassificationArtifactResources[CategoricalFeatureSpecProtocolT],
        ArtifactResultT,
        ArtifactHyperparamsT,
        CategoricalFeatureSpecProtocolT,
    ],
    Generic[
        ArtifactResultT,
        ArtifactHyperparamsT,
        CategoricalFeatureSpecProtocolT,
    ],
):
    @abstractmethod
    def _evaluate_classification(
        self, classification_results: ClassificationResults[CategoricalFeatureSpecProtocolT]
    ) -> ArtifactResultT: ...

    def _validate_classification_results(
        self, classification_results: ClassificationResults[CategoricalFeatureSpecProtocolT]
    ) -> ClassificationResults[CategoricalFeatureSpecProtocolT]:
        assert len(classification_results) > 0, (
            f"Expected nonempty classification results, got {classification_results.n_items=}"
        )
        return classification_results

    def _compute(
        self, resources: ClassificationArtifactResources[CategoricalFeatureSpecProtocolT]
    ) -> ArtifactResultT:
        result = self._evaluate_classification(
            classification_results=resources.classification_results
        )
        return result

    def _validate(
        self, resources: ClassificationArtifactResources[CategoricalFeatureSpecProtocolT]
    ) -> ClassificationArtifactResources[CategoricalFeatureSpecProtocolT]:
        classification_results = resources.classification_results
        classification_results_validated = self._validate_classification_results(
            classification_results=classification_results
        )
        resources_validated = ClassificationArtifactResources[CategoricalFeatureSpecProtocolT](
            classification_results=classification_results_validated
        )
        return resources_validated
