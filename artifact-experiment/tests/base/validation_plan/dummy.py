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


class DummyResourceSpec(ResourceSpecProtocol):
    pass


class DummyArtifactResources(ArtifactResources):
    pass


@DummyArtifactRegistry.register_artifact_config(artifact_type=DummyArtifactType.SCORE1)
class DummyArtifactHyperparams(ArtifactHyperparams):
    pass


@DummyArtifactRegistry.register_artifact(artifact_type=DummyArtifactType.SCORE1)
class DummyArtifact(
    Artifact[
        DummyArtifactResources,
        float,
        DummyArtifactHyperparams,
        DummyResourceSpec,
    ]
):
    def _compute(self, resources: DummyArtifactResources) -> float:
        _ = resources
        return 0.0

    def _validate(self, resources: DummyArtifactResources) -> DummyArtifactResources:
        return resources


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
        return [DummyArtifactType.SCORE1]

    @staticmethod
    def _get_array_types() -> List[DummyArtifactType]:
        return []

    @staticmethod
    def _get_plot_types() -> List[DummyArtifactType]:
        return []

    @staticmethod
    def _get_score_collection_types() -> List[DummyArtifactType]:
        return []

    @staticmethod
    def _get_array_collection_types() -> List[DummyArtifactType]:
        return []

    @staticmethod
    def _get_plot_collection_types() -> List[DummyArtifactType]:
        return []

    @staticmethod
    def _get_callback_factory() -> Type[DummyCallbackFactory]:
        return DummyCallbackFactory
