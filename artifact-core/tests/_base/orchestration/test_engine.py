from typing import Any, Callable, Tuple, Union
from unittest.mock import MagicMock

import numpy as np
import pytest
from artifact_core._base.typing.artifact_result import (
    Array,
    ArrayCollection,
    ArtifactResult,
    Plot,
    PlotCollection,
    ScoreCollection,
)
from matplotlib.figure import Figure
from pytest_mock import MockerFixture

from tests._base.dummy.artifacts.base import DummyArtifact
from tests._base.dummy.engine.engine import DummyArtifactEngine
from tests._base.dummy.registries.base import DummyArtifactRegistry
from tests._base.dummy.resource_spec import DummyResourceSpec
from tests._base.dummy.resources import DummyArtifactResources
from tests._base.dummy.types.array_collections import DummyArrayCollectionType
from tests._base.dummy.types.arrays import DummyArrayType
from tests._base.dummy.types.plot_collections import DummyPlotCollectionType
from tests._base.dummy.types.plots import DummyPlotType
from tests._base.dummy.types.score_collections import DummyScoreCollectionType
from tests._base.dummy.types.scores import DummyScoreType


@pytest.fixture
def engine_factory(
    mocker: MockerFixture,
) -> Callable[
    [str, DummyResourceSpec, ArtifactResult],
    Tuple[MagicMock, MagicMock, MagicMock, DummyArtifactEngine],
]:
    def _factory(
        modality: str,
        resource_spec: DummyResourceSpec,
        return_value: ArtifactResult,
    ) -> Tuple[MagicMock, MagicMock, MagicMock, DummyArtifactEngine]:
        artifact = mocker.Mock(name="artifact", spec_set=DummyArtifact[Any, Any])
        registry = mocker.Mock(name="registry", spec_set=DummyArtifactRegistry[Any, Any])
        other_registry = mocker.Mock(
            name="other_registry", spec_set=DummyArtifactRegistry[Any, Any]
        )
        artifact.compute.return_value = return_value
        registry.get.return_value = artifact
        registries = {
            f"{modality}_registry": other_registry
            for modality in [
                "score",
                "array",
                "plot",
                "score_collection",
                "array_collection",
                "plot_collection",
            ]
        }
        registries[f"{modality}_registry"] = registry
        engine = DummyArtifactEngine(resource_spec=resource_spec, **registries)
        return artifact, registry, other_registry, engine

    return _factory


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, resource_spec, resources, expected_result",
    [
        ("SCORE_TYPE_1", DummyResourceSpec(scale=1.0), DummyArtifactResources(valid=True, x=1), 1),
        ("SCORE_TYPE_2", DummyResourceSpec(scale=2.0), DummyArtifactResources(valid=True, x=2), 2),
        (
            "CUSTOM_SCORE_ARTIFACT",
            DummyResourceSpec(scale=3.0),
            DummyArtifactResources(valid=False, x=3),
            3,
        ),
    ],
)
def test_produce_score(
    engine_factory: Callable[
        [str, DummyResourceSpec, ArtifactResult],
        Tuple[MagicMock, MagicMock, MagicMock, DummyArtifactEngine],
    ],
    artifact_type: Union[DummyScoreType, str],
    resource_spec: DummyResourceSpec,
    resources: DummyArtifactResources,
    expected_result: float,
):
    artifact, registry, other_registry, engine = engine_factory(
        "score", resource_spec, expected_result
    )
    result = engine.produce_score(score_type=artifact_type, resources=resources)
    other_registry.get.assert_not_called()
    registry.get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    assert result == expected_result


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, resource_spec, resources, expected_result",
    [
        (
            "ARRAY_TYPE_1",
            DummyResourceSpec(scale=1.0),
            DummyArtifactResources(valid=True, x=1),
            np.array([1]),
        ),
        (
            "ARRAY_TYPE_2",
            DummyResourceSpec(scale=2.0),
            DummyArtifactResources(valid=True, x=10),
            np.array([10, 20]),
        ),
        (
            "CUSTOM_ARRAY",
            DummyResourceSpec(scale=3.0),
            DummyArtifactResources(valid=False, x=3),
            np.array([1, 2, 3]),
        ),
    ],
)
def test_produce_array(
    engine_factory: Callable[
        [str, DummyResourceSpec, ArtifactResult],
        Tuple[MagicMock, MagicMock, MagicMock, DummyArtifactEngine],
    ],
    artifact_type: Union[DummyArrayType, str],
    resource_spec: DummyResourceSpec,
    resources: DummyArtifactResources,
    expected_result: Array,
):
    artifact, registry, other_registry, engine = engine_factory(
        "array", resource_spec, expected_result
    )
    result = engine.produce_array(array_type=artifact_type, resources=resources)
    other_registry.get.assert_not_called()
    registry.get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    assert (result == expected_result).all()


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, resource_spec, resources, expected_result",
    [
        (
            "PLOT_TYPE_1",
            DummyResourceSpec(scale=1.0),
            DummyArtifactResources(valid=True, x=1),
            Figure(),
        ),
        (
            "PLOT_TYPE_2",
            DummyResourceSpec(scale=2.0),
            DummyArtifactResources(valid=True, x=10),
            Figure(),
        ),
        (
            "CUSTOM_PLOT",
            DummyResourceSpec(scale=3.0),
            DummyArtifactResources(valid=False, x=3),
            Figure(),
        ),
    ],
)
def test_produce_plot(
    engine_factory: Callable[
        [str, DummyResourceSpec, ArtifactResult],
        Tuple[MagicMock, MagicMock, MagicMock, DummyArtifactEngine],
    ],
    artifact_type: Union[DummyPlotType, str],
    resource_spec: DummyResourceSpec,
    resources: DummyArtifactResources,
    expected_result: Plot,
):
    artifact, registry, other_registry, engine = engine_factory(
        "plot", resource_spec, expected_result
    )
    result = engine.produce_plot(plot_type=artifact_type, resources=resources)
    other_registry.get.assert_not_called()
    registry.get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    assert result == expected_result


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, resource_spec, resources, expected_result",
    [
        (
            "SCORE_COLLECTION_TYPE_1",
            DummyResourceSpec(scale=1.0),
            DummyArtifactResources(valid=True, x=1),
            {"item": 1},
        ),
        (
            "SCORE_COLLECTION_TYPE_2",
            DummyResourceSpec(scale=2.0),
            DummyArtifactResources(valid=True, x=10),
            {"item": 2},
        ),
        (
            "CUSTOM_SCORE_COLLECTION",
            DummyResourceSpec(scale=3.0),
            DummyArtifactResources(valid=False, x=3),
            {"item": 3},
        ),
    ],
)
def test_produce_score_collection(
    engine_factory: Callable[
        [str, DummyResourceSpec, ArtifactResult],
        Tuple[MagicMock, MagicMock, MagicMock, DummyArtifactEngine],
    ],
    artifact_type: Union[DummyScoreCollectionType, str],
    resource_spec: DummyResourceSpec,
    resources: DummyArtifactResources,
    expected_result: ScoreCollection,
):
    artifact, registry, other_registry, engine = engine_factory(
        "score_collection", resource_spec, expected_result
    )
    result = engine.produce_score_collection(
        score_collection_type=artifact_type, resources=resources
    )
    other_registry.get.assert_not_called()
    registry.get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    assert result == expected_result


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, resource_spec, resources, expected_result",
    [
        (
            "ARRAY_COLLECTION_TYPE_1",
            DummyResourceSpec(scale=1.0),
            DummyArtifactResources(valid=True, x=1),
            {"item": np.array([1])},
        ),
        (
            "ARRAY_COLLECTION_TYPE_2",
            DummyResourceSpec(scale=2.0),
            DummyArtifactResources(valid=True, x=10),
            {"item": np.array([10])},
        ),
        (
            "CUSTOM_ARRAY_COLLECTION",
            DummyResourceSpec(scale=3.0),
            DummyArtifactResources(valid=False, x=3),
            {"item": np.array([1, 2, 3])},
        ),
    ],
)
def test_produce_array_collection(
    engine_factory: Callable[
        [str, DummyResourceSpec, ArtifactResult],
        Tuple[MagicMock, MagicMock, MagicMock, DummyArtifactEngine],
    ],
    artifact_type: Union[DummyArrayCollectionType, str],
    resource_spec: DummyResourceSpec,
    resources: DummyArtifactResources,
    expected_result: ArrayCollection,
):
    artifact, registry, other_registry, engine = engine_factory(
        "array_collection", resource_spec, expected_result
    )
    result = engine.produce_array_collection(
        array_collection_type=artifact_type, resources=resources
    )
    other_registry.get.assert_not_called()
    registry.get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    for array_name in result.keys():
        assert (result[array_name] == expected_result[array_name]).all()


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, resource_spec, resources, expected_result",
    [
        (
            "PLOT_COLLECTION_TYPE_1",
            DummyResourceSpec(scale=1.0),
            DummyArtifactResources(valid=True, x=1),
            {"item": Figure()},
        ),
        (
            "PLOT_COLLECTION_TYPE_2",
            DummyResourceSpec(scale=2.0),
            DummyArtifactResources(valid=True, x=10),
            {"item": Figure()},
        ),
        (
            "CUSTOM_PLOT_COLLECTION",
            DummyResourceSpec(scale=3.0),
            DummyArtifactResources(valid=False, x=3),
            {"item": Figure()},
        ),
    ],
)
def test_produce_plot_collection(
    engine_factory: Callable[
        [str, DummyResourceSpec, ArtifactResult],
        Tuple[MagicMock, MagicMock, MagicMock, DummyArtifactEngine],
    ],
    artifact_type: Union[DummyPlotCollectionType, str],
    resource_spec: DummyResourceSpec,
    resources: DummyArtifactResources,
    expected_result: PlotCollection,
):
    artifact, registry, other_registry, engine = engine_factory(
        "plot_collection", resource_spec, expected_result
    )
    result = engine.produce_plot_collection(plot_collection_type=artifact_type, resources=resources)
    other_registry.get.assert_not_called()
    registry.get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    assert result == expected_result
