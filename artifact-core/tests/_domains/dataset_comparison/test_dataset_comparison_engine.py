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
    Score,
    ScoreCollection,
)
from matplotlib.figure import Figure
from pytest_mock import MockerFixture

from tests._domains.dataset_comparison.dummy.artifacts import DummyDatasetComparisonArtifact
from tests._domains.dataset_comparison.dummy.engine import DummyDatasetComparisonEngine
from tests._domains.dataset_comparison.dummy.registries import (
    DummyDatasetComparisonArrayCollectionType,
    DummyDatasetComparisonArrayType,
    DummyDatasetComparisonPlotCollectionType,
    DummyDatasetComparisonPlotType,
    DummyDatasetComparisonRegistry,
    DummyDatasetComparisonScoreCollectionType,
    DummyDatasetComparisonScoreType,
)
from tests._domains.dataset_comparison.dummy.resource_spec import DummyResourceSpec
from tests._domains.dataset_comparison.dummy.resources import (
    DatasetComparisonArtifactResources,
    DummyDataset,
)


@pytest.fixture
def engine_factory(
    mocker: MockerFixture,
) -> Callable[
    [str, DummyResourceSpec, ArtifactResult],
    Tuple[MagicMock, MagicMock, MagicMock, DummyDatasetComparisonEngine],
]:
    def _factory(
        modality: str,
        resource_spec: DummyResourceSpec,
        return_value: ArtifactResult,
    ) -> Tuple[MagicMock, MagicMock, MagicMock, DummyDatasetComparisonEngine]:
        artifact = mocker.Mock(name="artifact", spec_set=DummyDatasetComparisonArtifact[Any, Any])
        registry = mocker.Mock(name="registry", spec_set=DummyDatasetComparisonRegistry[Any, Any])
        other_registry = mocker.Mock(
            name="other_registry", spec_set=DummyDatasetComparisonRegistry[Any, Any]
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
        engine = DummyDatasetComparisonEngine(resource_spec=resource_spec, **registries)
        return artifact, registry, other_registry, engine

    return _factory


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, resource_spec, dataset_real, dataset_synthetic, expected_result",
    [
        ("SCORE_TYPE_1", DummyResourceSpec(scale=1.0), DummyDataset(x=1), DummyDataset(x=1), 1),
        ("SCORE_TYPE_2", DummyResourceSpec(scale=2.0), DummyDataset(x=2), DummyDataset(x=2), 2),
        ("CUSTOM_SCORE", DummyResourceSpec(scale=3.0), DummyDataset(x=3), DummyDataset(x=3), 3),
    ],
)
def test_produce_dataset_comparison_score(
    engine_factory: Callable[
        [str, DummyResourceSpec, ArtifactResult],
        Tuple[MagicMock, MagicMock, MagicMock, DummyDatasetComparisonEngine],
    ],
    artifact_type: Union[DummyDatasetComparisonScoreType, str],
    resource_spec: DummyResourceSpec,
    dataset_real: DummyDataset,
    dataset_synthetic: DummyDataset,
    expected_result: Score,
):
    artifact, registry, other_registry, engine = engine_factory(
        "score", resource_spec, expected_result
    )
    resources = DatasetComparisonArtifactResources(
        dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
    )
    result = engine.produce_dataset_comparison_score(
        score_type=artifact_type, dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
    )
    other_registry.get.assert_not_called()
    registry.get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    assert result == expected_result


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, resource_spec, dataset_real, dataset_synthetic, expected_result",
    [
        (
            "ARRAY_TYPE_1",
            DummyResourceSpec(scale=1.0),
            DummyDataset(x=1),
            DummyDataset(x=1),
            np.array([1]),
        ),
        (
            "ARRAY_TYPE_2",
            DummyResourceSpec(scale=2.0),
            DummyDataset(x=2),
            DummyDataset(x=2),
            np.array([2]),
        ),
        (
            "CUSTOM_ARRAY",
            DummyResourceSpec(scale=3.0),
            DummyDataset(x=3),
            DummyDataset(x=3),
            np.array([3]),
        ),
    ],
)
def test_produce_dataset_comparison_array(
    engine_factory: Callable[
        [str, DummyResourceSpec, ArtifactResult],
        Tuple[MagicMock, MagicMock, MagicMock, DummyDatasetComparisonEngine],
    ],
    artifact_type: Union[DummyDatasetComparisonArrayType, str],
    resource_spec: DummyResourceSpec,
    dataset_real: DummyDataset,
    dataset_synthetic: DummyDataset,
    expected_result: Array,
):
    artifact, registry, other_registry, engine = engine_factory(
        "array", resource_spec, expected_result
    )
    resources = DatasetComparisonArtifactResources(
        dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
    )
    result = engine.produce_dataset_comparison_array(
        array_type=artifact_type, dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
    )
    other_registry.get.assert_not_called()
    registry.get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    assert (result == expected_result).all()


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, resource_spec, dataset_real, dataset_synthetic, expected_result",
    [
        (
            "PLOT_TYPE_1",
            DummyResourceSpec(scale=1.0),
            DummyDataset(x=1),
            DummyDataset(x=1),
            Figure(),
        ),
        (
            "PLOT_TYPE_2",
            DummyResourceSpec(scale=2.0),
            DummyDataset(x=2),
            DummyDataset(x=2),
            Figure(),
        ),
        (
            "CUSTOM_PLOT",
            DummyResourceSpec(scale=3.0),
            DummyDataset(x=3),
            DummyDataset(x=3),
            Figure(),
        ),
    ],
)
def test_produce_dataset_comparison_plot(
    engine_factory: Callable[
        [str, DummyResourceSpec, ArtifactResult],
        Tuple[MagicMock, MagicMock, MagicMock, DummyDatasetComparisonEngine],
    ],
    artifact_type: Union[DummyDatasetComparisonPlotType, str],
    resource_spec: DummyResourceSpec,
    dataset_real: DummyDataset,
    dataset_synthetic: DummyDataset,
    expected_result: Plot,
):
    artifact, registry, other_registry, engine = engine_factory(
        "plot", resource_spec, expected_result
    )
    resources = DatasetComparisonArtifactResources(
        dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
    )
    result = engine.produce_dataset_comparison_plot(
        plot_type=artifact_type, dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
    )
    other_registry.get.assert_not_called()
    registry.get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    assert result == expected_result


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, resource_spec, dataset_real, dataset_synthetic, expected_result",
    [
        (
            "SCORE_COLLECTION_TYPE_1",
            DummyResourceSpec(scale=1.0),
            DummyDataset(x=1),
            DummyDataset(x=1),
            {"item": 1},
        ),
        (
            "SCORE_COLLECTION_TYPE_2",
            DummyResourceSpec(scale=2.0),
            DummyDataset(x=2),
            DummyDataset(x=2),
            {"item": 2},
        ),
        (
            "CUSTOM_SCORE_COLLECTION",
            DummyResourceSpec(scale=3.0),
            DummyDataset(x=3),
            DummyDataset(x=3),
            {"item": 3},
        ),
    ],
)
def test_produce_dataset_comparison_score_collection(
    engine_factory: Callable[
        [str, DummyResourceSpec, ArtifactResult],
        Tuple[MagicMock, MagicMock, MagicMock, DummyDatasetComparisonEngine],
    ],
    artifact_type: Union[DummyDatasetComparisonScoreCollectionType, str],
    resource_spec: DummyResourceSpec,
    dataset_real: DummyDataset,
    dataset_synthetic: DummyDataset,
    expected_result: ScoreCollection,
):
    artifact, registry, other_registry, engine = engine_factory(
        "score_collection", resource_spec, expected_result
    )
    resources = DatasetComparisonArtifactResources(
        dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
    )
    result = engine.produce_dataset_comparison_score_collection(
        score_collection_type=artifact_type,
        dataset_real=dataset_real,
        dataset_synthetic=dataset_synthetic,
    )
    other_registry.get.assert_not_called()
    registry.get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    assert result == expected_result


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, resource_spec, dataset_real, dataset_synthetic, expected_result",
    [
        (
            "ARRAY_COLLECTION_TYPE_1",
            DummyResourceSpec(scale=1.0),
            DummyDataset(x=1),
            DummyDataset(x=1),
            {"item": np.array([1])},
        ),
        (
            "ARRAY_COLLECTION_TYPE_2",
            DummyResourceSpec(scale=2.0),
            DummyDataset(x=2),
            DummyDataset(x=2),
            {"item": np.array([2])},
        ),
        (
            "CUSTOM_ARRAY_COLLECTION",
            DummyResourceSpec(scale=3.0),
            DummyDataset(x=3),
            DummyDataset(x=3),
            {"item": np.array([3])},
        ),
    ],
)
def test_produce_dataset_comparison_array_collection(
    engine_factory: Callable[
        [str, DummyResourceSpec, ArtifactResult],
        Tuple[MagicMock, MagicMock, MagicMock, DummyDatasetComparisonEngine],
    ],
    artifact_type: Union[DummyDatasetComparisonArrayCollectionType, str],
    resource_spec: DummyResourceSpec,
    dataset_real: DummyDataset,
    dataset_synthetic: DummyDataset,
    expected_result: ArrayCollection,
):
    artifact, registry, other_registry, engine = engine_factory(
        "array_collection", resource_spec, expected_result
    )
    resources = DatasetComparisonArtifactResources(
        dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
    )
    result = engine.produce_dataset_comparison_array_collection(
        array_collection_type=artifact_type,
        dataset_real=dataset_real,
        dataset_synthetic=dataset_synthetic,
    )
    other_registry.get.assert_not_called()
    registry.get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    for array_name in result.keys():
        assert (result[array_name] == expected_result[array_name]).all()


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, resource_spec, dataset_real, dataset_synthetic, expected_result",
    [
        (
            "PLOT_COLLECTION_TYPE_1",
            DummyResourceSpec(scale=1.0),
            DummyDataset(x=1),
            DummyDataset(x=1),
            {"item": Figure()},
        ),
        (
            "PLOT_COLLECTION_TYPE_2",
            DummyResourceSpec(scale=2.0),
            DummyDataset(x=2),
            DummyDataset(x=2),
            {"item": Figure()},
        ),
        (
            "CUSTOM_PLOT_COLLECTION",
            DummyResourceSpec(scale=3.0),
            DummyDataset(x=3),
            DummyDataset(x=3),
            {"item": Figure()},
        ),
    ],
)
def test_produce_dataset_comparison_plot_collection(
    engine_factory: Callable[
        [str, DummyResourceSpec, ArtifactResult],
        Tuple[MagicMock, MagicMock, MagicMock, DummyDatasetComparisonEngine],
    ],
    artifact_type: Union[DummyDatasetComparisonPlotCollectionType, str],
    resource_spec: DummyResourceSpec,
    dataset_real: DummyDataset,
    dataset_synthetic: DummyDataset,
    expected_result: PlotCollection,
):
    artifact, registry, other_registry, engine = engine_factory(
        "plot_collection", resource_spec, expected_result
    )
    resources = DatasetComparisonArtifactResources(
        dataset_real=dataset_real, dataset_synthetic=dataset_synthetic
    )
    result = engine.produce_dataset_comparison_plot_collection(
        plot_collection_type=artifact_type,
        dataset_real=dataset_real,
        dataset_synthetic=dataset_synthetic,
    )
    other_registry.get.assert_not_called()
    registry.get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    artifact.compute.assert_called_once_with(resources=resources)
    assert result == expected_result
