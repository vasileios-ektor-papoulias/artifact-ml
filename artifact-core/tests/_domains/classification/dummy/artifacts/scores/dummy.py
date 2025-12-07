from dataclasses import dataclass

from artifact_core._base.core.hyperparams import ArtifactHyperparams

from tests._domains.classification.dummy.artifacts.base import DummyClassificationArtifact
from tests._domains.classification.dummy.registries.scores import DummyClassificationScoreRegistry
from tests._domains.classification.dummy.resources import (
    DummyClassificationResults,
    DummyClassStore,
)
from tests._domains.classification.dummy.types.scores import DummyClassificationScoreType


@DummyClassificationScoreRegistry.register_artifact_hyperparams(
    artifact_type=DummyClassificationScoreType.DUMMY_SCORE
)
@dataclass(frozen=True)
class DummyClassificationScoreHyperparams(ArtifactHyperparams):
    weight: float


@DummyClassificationScoreRegistry.register_artifact(
    artifact_type=DummyClassificationScoreType.DUMMY_SCORE
)
class DummyClassificationScore(
    DummyClassificationArtifact[DummyClassificationScoreHyperparams, float]
):
    def _evaluate_classification(
        self,
        true_class_store: DummyClassStore,
        classification_results: DummyClassificationResults,
    ) -> float:
        correct = 0
        total = 0
        for identifier in true_class_store.ids:
            true_idx = true_class_store.get_class_idx(identifier=identifier)
            pred_idx = classification_results.get_predicted_index(identifier=identifier)
            if true_idx == pred_idx:
                correct += 1
            total += 1
        if total == 0:
            return 0.0
        accuracy = correct / total
        result = accuracy * self._hyperparams.weight
        return result
