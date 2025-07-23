from typing import Any, Dict, List, Type

import numpy as np
from artifact_core.base.artifact import Artifact
from artifact_core.base.artifact_dependencies import (
    ArtifactHyperparams,
    ArtifactResources,
    ResourceSpecProtocol,
)
from artifact_core.base.registry import ArtifactRegistry, ArtifactType
from artifact_experiment.base.callbacks.factory import ArtifactCallbackFactory
from artifact_experiment.base.validation_plan import ValidationPlan
from matplotlib.figure import Figure


class DummyArtifactType(ArtifactType):
    SCORE1 = "score1"
    SCORE2 = "score2"
    ARRAY1 = "array1"
    PLOT1 = "plot1"
    SCORE_COLLECTION1 = "score_collection1"
    ARRAY_COLLECTION1 = "array_collection1"
    PLOT_COLLECTION1 = "plot_collection1"


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
class DummyScoreArtifact1(
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


@DummyArtifactRegistry.register_artifact_config(artifact_type=DummyArtifactType.SCORE2)
class DummyArtifactHyperparams2(ArtifactHyperparams):
    pass


@DummyArtifactRegistry.register_artifact(artifact_type=DummyArtifactType.SCORE2)
class DummyScoreArtifact2(
    Artifact[
        DummyArtifactResources,
        float,
        DummyArtifactHyperparams2,
        DummyResourceSpec,
    ]
):
    def _compute(self, resources: DummyArtifactResources) -> float:
        _ = resources
        return 1.0

    def _validate(self, resources: DummyArtifactResources) -> DummyArtifactResources:
        return resources


@DummyArtifactRegistry.register_artifact_config(artifact_type=DummyArtifactType.ARRAY1)
class DummyArrayHyperparams(ArtifactHyperparams):
    pass


@DummyArtifactRegistry.register_artifact(artifact_type=DummyArtifactType.ARRAY1)
class DummyArrayArtifact(
    Artifact[
        DummyArtifactResources,
        np.ndarray,
        DummyArrayHyperparams,
        DummyResourceSpec,
    ]
):
    def _compute(self, resources: DummyArtifactResources) -> np.ndarray:
        _ = resources
        return np.array([1, 2, 3])

    def _validate(self, resources: DummyArtifactResources) -> DummyArtifactResources:
        return resources


@DummyArtifactRegistry.register_artifact_config(artifact_type=DummyArtifactType.PLOT1)
class DummyPlotHyperparams(ArtifactHyperparams):
    pass


@DummyArtifactRegistry.register_artifact(artifact_type=DummyArtifactType.PLOT1)
class DummyPlotArtifact(
    Artifact[
        DummyArtifactResources,
        Figure,
        DummyPlotHyperparams,
        DummyResourceSpec,
    ]
):
    def _compute(self, resources: DummyArtifactResources) -> Figure:
        _ = resources
        return Figure()

    def _validate(self, resources: DummyArtifactResources) -> DummyArtifactResources:
        return resources


@DummyArtifactRegistry.register_artifact_config(artifact_type=DummyArtifactType.SCORE_COLLECTION1)
class DummyScoreCollectionHyperparams(ArtifactHyperparams):
    pass


@DummyArtifactRegistry.register_artifact(artifact_type=DummyArtifactType.SCORE_COLLECTION1)
class DummyScoreCollectionArtifact(
    Artifact[
        DummyArtifactResources,
        Dict[str, float],
        DummyScoreCollectionHyperparams,
        DummyResourceSpec,
    ]
):
    def _compute(self, resources: DummyArtifactResources) -> Dict[str, float]:
        _ = resources
        return {"score1": 1.0, "score2": 2.0}

    def _validate(self, resources: DummyArtifactResources) -> DummyArtifactResources:
        return resources


@DummyArtifactRegistry.register_artifact_config(artifact_type=DummyArtifactType.ARRAY_COLLECTION1)
class DummyArrayCollectionHyperparams(ArtifactHyperparams):
    pass


@DummyArtifactRegistry.register_artifact(artifact_type=DummyArtifactType.ARRAY_COLLECTION1)
class DummyArrayCollectionArtifact(
    Artifact[
        DummyArtifactResources,
        Dict[str, np.ndarray],
        DummyArrayCollectionHyperparams,
        DummyResourceSpec,
    ]
):
    def _compute(self, resources: DummyArtifactResources) -> Dict[str, np.ndarray]:
        _ = resources
        return {"array1": np.array([1, 2]), "array2": np.array([3, 4])}

    def _validate(self, resources: DummyArtifactResources) -> DummyArtifactResources:
        return resources


@DummyArtifactRegistry.register_artifact_config(artifact_type=DummyArtifactType.PLOT_COLLECTION1)
class DummyPlotCollectionHyperparams(ArtifactHyperparams):
    pass


@DummyArtifactRegistry.register_artifact(artifact_type=DummyArtifactType.PLOT_COLLECTION1)
class DummyPlotCollectionArtifact(
    Artifact[
        DummyArtifactResources,
        Dict[str, Figure],
        DummyPlotCollectionHyperparams,
        DummyResourceSpec,
    ]
):
    def _compute(self, resources: DummyArtifactResources) -> Dict[str, Figure]:
        _ = resources
        return {"plot1": Figure(), "plot2": Figure()}

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
