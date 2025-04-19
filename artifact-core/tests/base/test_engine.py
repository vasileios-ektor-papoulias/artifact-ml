from typing import Callable, Dict, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest
from artifact_core.base.artifact_dependencies import ArtifactResult
from matplotlib.figure import Figure
from pytest_mock import MockerFixture

from tests.base.dummy.artifact_dependencies import DummyArtifactResources, DummyDataSpec
from tests.base.dummy.engine import DummyArtifactEngine


@pytest.fixture
def mock_dependency_factory(
    mocker: MockerFixture,
) -> Callable[[ArtifactResult], Tuple[MagicMock, MagicMock, MagicMock]]:
    def factory(return_value: ArtifactResult) -> Tuple[MagicMock, MagicMock, MagicMock]:
        artifact_type = mocker.Mock(name="dummy_artifact_type")
        artifact = mocker.Mock(name="dummy_artifact")
        artifact.compute.return_value = return_value

        registry = mocker.Mock(name="dummy_registry")
        registry.get.return_value = artifact

        return artifact_type, artifact, registry

    return factory


@pytest.mark.parametrize(
    "data_spec, resources, expected_result",
    [
        (DummyDataSpec(scale=1), DummyArtifactResources(valid=True, x=1), 1),
        (DummyDataSpec(scale=1), DummyArtifactResources(valid=True, x=10), 2),
        (DummyDataSpec(scale=1), DummyArtifactResources(valid=False, x=1), 3),
    ],
)
def test_produce_score(
    mock_dependency_factory: Callable[[ArtifactResult], Tuple[MagicMock, MagicMock, MagicMock]],
    data_spec: DummyDataSpec,
    resources: DummyArtifactResources,
    expected_result: float,
):
    artifact_type, artifact, registry = mock_dependency_factory(expected_result)

    engine = DummyArtifactEngine(data_spec=data_spec)
    engine._score_registry = registry  # type: ignore
    result = engine.produce_score(score_type=artifact_type, resources=resources)

    registry.get.assert_called_once_with(artifact_type=artifact_type, data_spec=data_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    assert result == expected_result


@pytest.mark.parametrize(
    "data_spec, resources, expected_result",
    [
        (DummyDataSpec(scale=1), DummyArtifactResources(valid=True, x=1), np.array([1])),
        (DummyDataSpec(scale=1), DummyArtifactResources(valid=True, x=10), np.array([10, 20])),
        (DummyDataSpec(scale=1), DummyArtifactResources(valid=False, x=1), np.array([1])),
    ],
)
def test_produce_array(
    mock_dependency_factory: Callable[[ArtifactResult], Tuple[MagicMock, MagicMock, MagicMock]],
    data_spec: DummyDataSpec,
    resources: DummyArtifactResources,
    expected_result: np.ndarray,
):
    artifact_type, artifact, registry = mock_dependency_factory(expected_result)

    engine = DummyArtifactEngine(data_spec=data_spec)
    engine._array_registry = registry  # type: ignore
    result = engine.produce_array(array_type=artifact_type, resources=resources)

    registry.get.assert_called_once_with(artifact_type=artifact_type, data_spec=data_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    assert (result == expected_result).all()


@pytest.mark.parametrize(
    "data_spec, resources, expected_result",
    [
        (DummyDataSpec(scale=1), DummyArtifactResources(valid=True, x=1), Figure()),
        (DummyDataSpec(scale=1), DummyArtifactResources(valid=True, x=10), Figure()),
        (DummyDataSpec(scale=1), DummyArtifactResources(valid=False, x=1), Figure()),
    ],
)
def test_produce_plot(
    mock_dependency_factory: Callable[[ArtifactResult], Tuple[MagicMock, MagicMock, MagicMock]],
    data_spec: DummyDataSpec,
    resources: DummyArtifactResources,
    expected_result: Figure,
):
    artifact_type, artifact, registry = mock_dependency_factory(expected_result)

    engine = DummyArtifactEngine(data_spec=data_spec)
    engine._plot_registry = registry  # type: ignore
    result = engine.produce_plot(plot_type=artifact_type, resources=resources)

    registry.get.assert_called_once_with(artifact_type=artifact_type, data_spec=data_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    assert result == expected_result


@pytest.mark.parametrize(
    "data_spec, resources, expected_result",
    [
        (DummyDataSpec(scale=1), DummyArtifactResources(valid=True, x=1), {"item": 1}),
        (DummyDataSpec(scale=1), DummyArtifactResources(valid=True, x=10), {"item": 2}),
        (DummyDataSpec(scale=1), DummyArtifactResources(valid=False, x=1), {"item": 3}),
    ],
)
def test_produce_score_collection(
    mock_dependency_factory: Callable[[ArtifactResult], Tuple[MagicMock, MagicMock, MagicMock]],
    data_spec: DummyDataSpec,
    resources: DummyArtifactResources,
    expected_result: Dict[str, float],
):
    artifact_type, artifact, registry = mock_dependency_factory(expected_result)

    engine = DummyArtifactEngine(data_spec=data_spec)
    engine._score_collection_registry = registry  # type: ignore
    result = engine.produce_score_collection(
        score_collection_type=artifact_type, resources=resources
    )

    registry.get.assert_called_once_with(artifact_type=artifact_type, data_spec=data_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    assert result == expected_result


@pytest.mark.parametrize(
    "data_spec, resources, expected_result",
    [
        (DummyDataSpec(scale=1), DummyArtifactResources(valid=True, x=1), {"item": np.array([1])}),
        (
            DummyDataSpec(scale=1),
            DummyArtifactResources(valid=True, x=10),
            {"item": np.array([10])},
        ),
        (DummyDataSpec(scale=1), DummyArtifactResources(valid=False, x=1), {"item": np.array([1])}),
    ],
)
def test_produce_array_collection(
    mock_dependency_factory: Callable[[ArtifactResult], Tuple[MagicMock, MagicMock, MagicMock]],
    data_spec: DummyDataSpec,
    resources: DummyArtifactResources,
    expected_result: Dict[str, np.ndarray],
):
    artifact_type, artifact, registry = mock_dependency_factory(expected_result)

    engine = DummyArtifactEngine(data_spec=data_spec)
    engine._array_collection_registry = registry  # type: ignore
    result = engine.produce_array_collection(
        array_collection_type=artifact_type, resources=resources
    )

    registry.get.assert_called_once_with(artifact_type=artifact_type, data_spec=data_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    for array_name in result.keys():
        assert (result[array_name] == expected_result[array_name]).all()


@pytest.mark.parametrize(
    "data_spec, resources, expected_result",
    [
        (DummyDataSpec(scale=1), DummyArtifactResources(valid=True, x=1), {"item": Figure()}),
        (DummyDataSpec(scale=1), DummyArtifactResources(valid=True, x=10), {"item": Figure()}),
        (DummyDataSpec(scale=1), DummyArtifactResources(valid=False, x=1), {"item": Figure()}),
    ],
)
def test_produce_plot_collection(
    mock_dependency_factory: Callable[[ArtifactResult], Tuple[MagicMock, MagicMock, MagicMock]],
    data_spec: DummyDataSpec,
    resources: DummyArtifactResources,
    expected_result: Dict[str, Figure],
):
    artifact_type, artifact, registry = mock_dependency_factory(expected_result)

    engine = DummyArtifactEngine(data_spec=data_spec)
    engine._plot_collection_registry = registry  # type: ignore
    result = engine.produce_plot_collection(plot_collection_type=artifact_type, resources=resources)

    registry.get.assert_called_once_with(artifact_type=artifact_type, data_spec=data_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    assert result == expected_result
