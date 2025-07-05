from typing import Any, Dict, List, Type

from artifact_core.base.artifact import Artifact
from artifact_core.base.artifact_dependencies import (
    ArtifactHyperparams,
    ArtifactResources,
    ResourceSpecProtocol,
)
from artifact_core.base.registry import ArtifactRegistry, ArtifactType
from artifact_experiment.base.callbacks.factory import ArtifactCallbackFactory
from artifact_experiment.base.validation_plan import ValidationPlan


class DummyArtifactType(ArtifactType):
    SCORE1 = "score1"
    SCORE2 = "score2"
    ARRAY1 = "array1"
    PLOT1 = "plot1"
    SCORE_COLLECTION1 = "score_collection1"
    ARRAY_COLLECTION1 = "array_collection1"
    PLOT_COLLECTION1 = "plot_collection1"


class DummyResourceSpec(ResourceSpecProtocol):
    pass


class DummyArtifactResources(ArtifactResources):
    pass


class DummyArtifactHyperparams(ArtifactHyperparams):
    pass


class DummyArtifact(
    Artifact[
        DummyArtifactResources,
        float,
        DummyArtifactHyperparams,
        DummyResourceSpec,
    ]
):
    def _compute(self, resources: DummyArtifactResources) -> float:
        return 42.0

    def _validate(self, resources: DummyArtifactResources) -> DummyArtifactResources:
        return resources


class DummyArtifactRegistry(ArtifactRegistry):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "SCORE1": {},
            "SCORE2": {},
            "ARRAY1": {},
            "PLOT1": {},
            "SCORE_COLLECTION1": {},
            "ARRAY_COLLECTION1": {},
            "PLOT_COLLECTION1": {},
        }

    @classmethod
    def _get_artifact_class(cls, artifact_type: str) -> Type[Artifact]:
        return DummyArtifact

    @classmethod
    def _get_artifact_hyperparams_class(cls, artifact_type: str):
        return DummyArtifactHyperparams


class DummyCallbackFactory(
    ArtifactCallbackFactory[
        DummyArtifactType,
        DummyArtifactType,
        DummyArtifactType,
        DummyArtifactType,
        DummyArtifactType,
        DummyArtifactType,
        DummyArtifactResources,
        DummyResourceSpec,
    ]
):
    @staticmethod
    def _get_score_registry() -> Type[ArtifactRegistry]:
        return DummyArtifactRegistry

    @staticmethod
    def _get_array_registry() -> Type[ArtifactRegistry]:
        return DummyArtifactRegistry

    @staticmethod
    def _get_plot_registry() -> Type[ArtifactRegistry]:
        return DummyArtifactRegistry

    @staticmethod
    def _get_score_collection_registry() -> Type[ArtifactRegistry]:
        return DummyArtifactRegistry

    @staticmethod
    def _get_array_collection_registry() -> Type[ArtifactRegistry]:
        return DummyArtifactRegistry

    @staticmethod
    def _get_plot_collection_registry() -> Type[ArtifactRegistry]:
        return DummyArtifactRegistry


class DummyValidationPlan(
    ValidationPlan[
        DummyArtifactType,
        DummyArtifactType,
        DummyArtifactType,
        DummyArtifactType,
        DummyArtifactType,
        DummyArtifactType,
        DummyArtifactResources,
        DummyResourceSpec,
    ]
):
    @staticmethod
    def _get_score_types() -> List[DummyArtifactType]:
        return [DummyArtifactType.SCORE1, DummyArtifactType.SCORE2]

    @staticmethod
    def _get_array_types() -> List[DummyArtifactType]:
        return [DummyArtifactType.ARRAY1]

    @staticmethod
    def _get_plot_types() -> List[DummyArtifactType]:
        return [DummyArtifactType.PLOT1]

    @staticmethod
    def _get_score_collection_types() -> List[DummyArtifactType]:
        return [DummyArtifactType.SCORE_COLLECTION1]

    @staticmethod
    def _get_array_collection_types() -> List[DummyArtifactType]:
        return [DummyArtifactType.ARRAY_COLLECTION1]

    @staticmethod
    def _get_plot_collection_types() -> List[DummyArtifactType]:
        return [DummyArtifactType.PLOT_COLLECTION1]

    @staticmethod
    def _get_callback_factory() -> Type[DummyCallbackFactory]:
        return DummyCallbackFactory
