from typing import Callable, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest
from artifact_core.base.artifact_dependencies import ArtifactResult
from matplotlib.figure import Figure
from pytest_mock import MockerFixture

from tests.core.dataset_comparison.dummy.artifact_dependencies import (
    DatasetComparisonArtifactResources,
    DummyDataset,
    DummyResourceSpec,
)
from tests.core.dataset_comparison.dummy.engine import DummyDatasetComparisonEngine


@pytest.fixture
def mock_dependency_factory(
    mocker: MockerFixture,
) -> Callable[[ArtifactResult], Tuple[MagicMock, MagicMock, MagicMock]]:
    def _factory(return_value: ArtifactResult) -> Tuple[MagicMock, MagicMock, MagicMock]:
        artifact_type = mocker.Mock(name="dummy_artifact_type")
        artifact = mocker.Mock(name="dummy_artifact")
        artifact.compute.return_value = return_value

        registry = mocker.Mock(name="dummy_registry")
        registry.get.return_value = artifact

        return artifact_type, artifact, registry

    return _factory


@pytest.mark.parametrize(
    "resource_spec, dataset_real, dataset_synthetic, expected_result",
    [
        (
            DummyResourceSpec(scale=1),
            DummyDataset(x=1),
            DummyDataset(x=1),
            1,
        ),
        (
            DummyResourceSpec(scale=2),
            DummyDataset(x=2),
            DummyDataset(x=2),
            2,
        ),
    ],
)
def test_produce_dataset_comparison_score(
    mock_dependency_factory: Callable[[ArtifactResult], Tuple[MagicMock, MagicMock, MagicMock]],
    resource_spec: DummyResourceSpec,
    dataset_real: DummyDataset,
    dataset_synthetic: DummyDataset,
    expected_result: float,
):
    artifact_type, artifact, registry = mock_dependency_factory(expected_result)

    engine = DummyDatasetComparisonEngine(resource_spec=resource_spec)
    engine._score_registry = registry  # type: ignore
    result = engine.produce_dataset_comparison_score(
        score_type=artifact_type, dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
    )
    resources = DatasetComparisonArtifactResources(
        dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
    )
    registry.get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    assert result == expected_result


@pytest.mark.parametrize(
    "resource_spec, dataset_real, dataset_synthetic, expected_result",
    [
        (
            DummyResourceSpec(scale=1),
            DummyDataset(x=1),
            DummyDataset(x=1),
            np.array([1]),
        ),
        (DummyResourceSpec(scale=2), DummyDataset(x=2), DummyDataset(x=2), np.array([2])),
    ],
)
def test_produce_dataset_comparison_array(
    mock_dependency_factory: Callable[[ArtifactResult], Tuple[MagicMock, MagicMock, MagicMock]],
    resource_spec: DummyResourceSpec,
    dataset_real: DummyDataset,
    dataset_synthetic: DummyDataset,
    expected_result: float,
):
    artifact_type, artifact, registry = mock_dependency_factory(expected_result)

    engine = DummyDatasetComparisonEngine(resource_spec=resource_spec)
    engine._array_registry = registry  # type: ignore
    result = engine.produce_dataset_comparison_array(
        array_type=artifact_type, dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
    )
    resources = DatasetComparisonArtifactResources(
        dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
    )
    registry.get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    assert result == expected_result


@pytest.mark.parametrize(
    "resource_spec, dataset_real, dataset_synthetic, expected_result",
    [
        (
            DummyResourceSpec(scale=1),
            DummyDataset(x=1),
            DummyDataset(x=1),
            Figure(),
        ),
        (DummyResourceSpec(scale=2), DummyDataset(x=2), DummyDataset(x=2), Figure()),
    ],
)
def test_produce_dataset_comparison_plot(
    mock_dependency_factory: Callable[[ArtifactResult], Tuple[MagicMock, MagicMock, MagicMock]],
    resource_spec: DummyResourceSpec,
    dataset_real: DummyDataset,
    dataset_synthetic: DummyDataset,
    expected_result: float,
):
    artifact_type, artifact, registry = mock_dependency_factory(expected_result)

    engine = DummyDatasetComparisonEngine(resource_spec=resource_spec)
    engine._plot_registry = registry  # type: ignore
    result = engine.produce_dataset_comparison_plot(
        plot_type=artifact_type, dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
    )
    resources = DatasetComparisonArtifactResources(
        dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
    )
    registry.get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    assert result == expected_result


@pytest.mark.parametrize(
    "resource_spec, dataset_real, dataset_synthetic, expected_result",
    [
        (
            DummyResourceSpec(scale=1),
            DummyDataset(x=1),
            DummyDataset(x=1),
            {"item": 1},
        ),
        (
            DummyResourceSpec(scale=2),
            DummyDataset(x=2),
            DummyDataset(x=2),
            {"item": 2},
        ),
    ],
)
def test_produce_dataset_comparison_score_collection(
    mock_dependency_factory: Callable[[ArtifactResult], Tuple[MagicMock, MagicMock, MagicMock]],
    resource_spec: DummyResourceSpec,
    dataset_real: DummyDataset,
    dataset_synthetic: DummyDataset,
    expected_result: float,
):
    artifact_type, artifact, registry = mock_dependency_factory(expected_result)

    engine = DummyDatasetComparisonEngine(resource_spec=resource_spec)
    engine._score_collection_registry = registry  # type: ignore
    result = engine.produce_dataset_comparison_score_collection(
        score_collection_type=artifact_type,
        dataset_real=dataset_real,
        dataset_synthetic=dataset_synthetic,
    )
    resources = DatasetComparisonArtifactResources(
        dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
    )
    registry.get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    assert result == expected_result


@pytest.mark.parametrize(
    "resource_spec, dataset_real, dataset_synthetic, expected_result",
    [
        (
            DummyResourceSpec(scale=1),
            DummyDataset(x=1),
            DummyDataset(x=1),
            {"ïtem": np.array([1])},
        ),
        (DummyResourceSpec(scale=2), DummyDataset(x=2), DummyDataset(x=2), {"item": np.array([2])}),
    ],
)
def test_produce_dataset_comparison_array_collection(
    mock_dependency_factory: Callable[[ArtifactResult], Tuple[MagicMock, MagicMock, MagicMock]],
    resource_spec: DummyResourceSpec,
    dataset_real: DummyDataset,
    dataset_synthetic: DummyDataset,
    expected_result: float,
):
    artifact_type, artifact, registry = mock_dependency_factory(expected_result)

    engine = DummyDatasetComparisonEngine(resource_spec=resource_spec)
    engine._array_collection_registry = registry  # type: ignore
    result = engine.produce_dataset_comparison_array_collection(
        array_collection_type=artifact_type,
        dataset_real=dataset_real,
        dataset_synthetic=dataset_synthetic,
    )
    resources = DatasetComparisonArtifactResources(
        dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
    )
    registry.get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    assert result == expected_result


@pytest.mark.parametrize(
    "resource_spec, dataset_real, dataset_synthetic, expected_result",
    [
        (
            DummyResourceSpec(scale=1),
            DummyDataset(x=1),
            DummyDataset(x=1),
            {"item": Figure()},
        ),
        (DummyResourceSpec(scale=2), DummyDataset(x=2), DummyDataset(x=2), {"item": Figure()}),
    ],
)
def test_produce_dataset_comparison_plot_collection(
    mock_dependency_factory: Callable[[ArtifactResult], Tuple[MagicMock, MagicMock, MagicMock]],
    resource_spec: DummyResourceSpec,
    dataset_real: DummyDataset,
    dataset_synthetic: DummyDataset,
    expected_result: float,
):
    artifact_type, artifact, registry = mock_dependency_factory(expected_result)

    engine = DummyDatasetComparisonEngine(resource_spec=resource_spec)
    engine._plot_collection_registry = registry  # type: ignore
    result = engine.produce_dataset_comparison_plot_collection(
        plot_collection_type=artifact_type,
        dataset_real=dataset_real,
        dataset_synthetic=dataset_synthetic,
    )
    resources = DatasetComparisonArtifactResources(
        dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
    )
    registry.get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    assert result == expected_result
