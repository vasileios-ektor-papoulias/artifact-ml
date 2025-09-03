from typing import Dict, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from artifact_core.base.artifact import Artifact
from artifact_core.base.artifact_dependencies import (
    ArtifactResult,
    NoArtifactHyperparams,
)
from matplotlib.figure import Figure
from numpy import ndarray

from tests.base.dummy_artifact_toolkit.registries import (
    DummyArrayCollectionRegistry,
    DummyArrayCollectionType,
    DummyArrayRegistry,
    DummyArrayType,
    DummyArtifactResources,
    DummyPlotCollectionRegistry,
    DummyPlotCollectionType,
    DummyPlotRegistry,
    DummyPlotType,
    DummyResourceSpec,
    DummyScoreCollectionRegistry,
    DummyScoreCollectionType,
    DummyScoreRegistry,
    DummyScoreType,
)

artifactResultT = TypeVar("artifactResultT", bound=ArtifactResult)


class DummyArtifact(
    Artifact[DummyArtifactResources, artifactResultT, NoArtifactHyperparams, DummyResourceSpec]
): ...


@DummyScoreRegistry.register_artifact(artifact_type=DummyScoreType.DUMMY_SCORE_1)
class DummyScore1(DummyArtifact[float]):
    def _validate(self, resources: DummyArtifactResources) -> DummyArtifactResources:
        return resources

    def _compute(self, resources: DummyArtifactResources) -> float:
        _ = resources
        return 0


@DummyArrayRegistry.register_artifact(artifact_type=DummyArrayType.DUMMY_ARRAY_1)
class DummyArray1(DummyArtifact[ndarray]):
    def _validate(self, resources: DummyArtifactResources) -> DummyArtifactResources:
        return resources

    def _compute(self, resources: DummyArtifactResources) -> ndarray:
        _ = resources
        return np.array([1.0, 2.0, 3.0])


@DummyPlotRegistry.register_artifact(artifact_type=DummyPlotType.DUMMY_PLOT_1)
class DummyPlot1(DummyArtifact[Figure]):
    def _validate(self, resources: DummyArtifactResources) -> DummyArtifactResources:
        return resources

    def _compute(self, resources: DummyArtifactResources) -> Figure:
        _ = resources
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        return fig


@DummyScoreCollectionRegistry.register_artifact(
    artifact_type=DummyScoreCollectionType.DUMMY_SCORE_COLLECTION_1
)
class DummyScoreCollection1(DummyArtifact[Dict[str, float]]):
    def _validate(self, resources: DummyArtifactResources) -> DummyArtifactResources:
        return resources

    def _compute(self, resources: DummyArtifactResources) -> Dict[str, float]:
        _ = resources
        return {"score_1": 0.1, "score_2": 0.2}


@DummyArrayCollectionRegistry.register_artifact(
    artifact_type=DummyArrayCollectionType.DUMMY_ARRAY_COLLECTION_1
)
class DummyArrayCollection1(DummyArtifact[Dict[str, ndarray]]):
    def _validate(self, resources: DummyArtifactResources) -> DummyArtifactResources:
        return resources

    def _compute(self, resources: DummyArtifactResources) -> Dict[str, ndarray]:
        _ = resources
        return {"array_1": np.array([1, 2]), "array_2": np.array([3, 4])}


@DummyPlotCollectionRegistry.register_artifact(
    artifact_type=DummyPlotCollectionType.DUMMY_PLOT_COLLECTION_1
)
class DummyPlotCollection1(DummyArtifact[Dict[str, Figure]]):
    def _validate(self, resources: DummyArtifactResources) -> DummyArtifactResources:
        return resources

    def _compute(self, resources: DummyArtifactResources) -> Dict[str, Figure]:
        _ = resources
        fig1, ax1 = plt.subplots()
        ax1.plot([1, 2], [3, 4])
        fig2, ax2 = plt.subplots()
        ax2.plot([0, 1], [1, 0])
        return {"plot_1": fig1, "plot_2": fig2}
