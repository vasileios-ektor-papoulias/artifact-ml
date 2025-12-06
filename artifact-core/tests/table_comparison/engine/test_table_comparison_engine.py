from typing import Any, Dict, Type
from unittest.mock import ANY

import numpy as np
import pandas as pd
import pytest
from artifact_core._domains.dataset_comparison.artifact import DatasetComparisonArtifactResources
from artifact_core._libs.resource_specs.table_comparison.protocol import TabularDataSpecProtocol
from artifact_core.table_comparison._artifacts.array_collections.descriptive_stats import (
    FirstQuartileJuxtapositionArrays,
    MaxJuxtapositionArrays,
    MeanJuxtapositionArrays,
    MedianJuxtapositionArrays,
    MinJuxtapositionArrays,
    STDJuxtapositionArrays,
    ThirdQuartileJuxtapositionArrays,
    VarianceJuxtapositionArrays,
)
from artifact_core.table_comparison._artifacts.base import TableComparisonArtifact
from artifact_core.table_comparison._artifacts.plot_collections.cdf import CDFPlots
from artifact_core.table_comparison._artifacts.plot_collections.correlations import (
    CorrelationHeatmaps,
)
from artifact_core.table_comparison._artifacts.plot_collections.pdf import PDFPlots
from artifact_core.table_comparison._artifacts.plots.cdf import CDFPlot
from artifact_core.table_comparison._artifacts.plots.correlations import (
    CorrelationHeatmapJuxtapositionPlot,
)
from artifact_core.table_comparison._artifacts.plots.descriptive_stats import (
    DescriptiveStatsAlignmentPlot,
    FirstQuartileAlignmentPlot,
    MaxAlignmentPlot,
    MeanAlignmentPlot,
    MedianAlignmentPlot,
    MinAlignmentPlot,
    STDAlignmentPlot,
    ThirdQuartileAlignmentPlot,
    VarianceAlignmentPlot,
)
from artifact_core.table_comparison._artifacts.plots.pca import PCAJuxtapositionPlot
from artifact_core.table_comparison._artifacts.plots.pdf import PDFPlot
from artifact_core.table_comparison._artifacts.plots.truncated_svd import (
    TruncatedSVDJuxtapositionPlot,
)
from artifact_core.table_comparison._artifacts.plots.tsne import TSNEJuxtapositionPlot
from artifact_core.table_comparison._artifacts.score_collections.js import JSDistanceScores
from artifact_core.table_comparison._artifacts.scores.correlation import CorrelationDistanceScore
from artifact_core.table_comparison._artifacts.scores.mean_js import MeanJSDistanceScore
from artifact_core.table_comparison._engine.engine import (
    TableComparisonArrayCollectionRegistry,
    TableComparisonArrayRegistry,
    TableComparisonEngine,
    TableComparisonPlotCollectionRegistry,
    TableComparisonPlotRegistry,
    TableComparisonScoreCollectionRegistry,
    TableComparisonScoreRegistry,
)
from artifact_core.table_comparison._types.array_collections import (
    TableComparisonArrayCollectionType,
)
from artifact_core.table_comparison._types.arrays import TableComparisonArrayType
from artifact_core.table_comparison._types.plot_collections import TableComparisonPlotCollectionType
from artifact_core.table_comparison._types.plots import TableComparisonPlotType
from artifact_core.table_comparison._types.score_collections import (
    TableComparisonScoreCollectionType,
)
from artifact_core.table_comparison._types.scores import TableComparisonScoreType
from matplotlib.figure import Figure
from pytest_mock import MockerFixture

SCORE_TEST_CASES = [
    (TableComparisonScoreType.MEAN_JS_DISTANCE, MeanJSDistanceScore, 0.123),
    (TableComparisonScoreType.CORRELATION_DISTANCE, CorrelationDistanceScore, 0.456),
]

ARRAY_TEST_CASES: list[tuple[TableComparisonArrayType, type, Any]] = []

PLOT_TEST_CASES = [
    (TableComparisonPlotType.PDF, PDFPlot),
    (TableComparisonPlotType.CDF, CDFPlot),
    (TableComparisonPlotType.DESCRIPTIVE_STATS_ALIGNMENT, DescriptiveStatsAlignmentPlot),
    (TableComparisonPlotType.MEAN_ALIGNMENT, MeanAlignmentPlot),
    (TableComparisonPlotType.STD_ALIGNMENT, STDAlignmentPlot),
    (TableComparisonPlotType.VARIANCE_ALIGNMENT, VarianceAlignmentPlot),
    (TableComparisonPlotType.MEDIAN_ALIGNMENT, MedianAlignmentPlot),
    (TableComparisonPlotType.FIRST_QUARTILE_ALIGNMENT, FirstQuartileAlignmentPlot),
    (TableComparisonPlotType.THIRD_QUARTILE_ALIGNMENT, ThirdQuartileAlignmentPlot),
    (TableComparisonPlotType.MIN_ALIGNMENT, MinAlignmentPlot),
    (TableComparisonPlotType.MAX_ALIGNMENT, MaxAlignmentPlot),
    (
        TableComparisonPlotType.CORRELATION_HEATMAP_JUXTAPOSITION,
        CorrelationHeatmapJuxtapositionPlot,
    ),
    (TableComparisonPlotType.PCA_JUXTAPOSITION, PCAJuxtapositionPlot),
    (TableComparisonPlotType.TRUNCATED_SVD_JUXTAPOSITION, TruncatedSVDJuxtapositionPlot),
    (TableComparisonPlotType.TSNE_JUXTAPOSITION, TSNEJuxtapositionPlot),
]

SCORE_COLLECTION_TEST_CASES = [
    (TableComparisonScoreCollectionType.JS_DISTANCE, JSDistanceScores, {"a": 0.1, "b": 0.2}),
]

ARRAY_COLLECTION_TEST_CASES = [
    (TableComparisonArrayCollectionType.MEAN_JUXTAPOSITION, MeanJuxtapositionArrays),
    (TableComparisonArrayCollectionType.STD_JUXTAPOSITION, STDJuxtapositionArrays),
    (TableComparisonArrayCollectionType.VARIANCE_JUXTAPOSITION, VarianceJuxtapositionArrays),
    (TableComparisonArrayCollectionType.MEDIAN_JUXTAPOSITION, MedianJuxtapositionArrays),
    (
        TableComparisonArrayCollectionType.FIRST_QUARTILE_JUXTAPOSITION,
        FirstQuartileJuxtapositionArrays,
    ),
    (
        TableComparisonArrayCollectionType.THIRD_QUARTILE_JUXTAPOSITION,
        ThirdQuartileJuxtapositionArrays,
    ),
    (TableComparisonArrayCollectionType.MIN_JUXTAPOSITION, MinJuxtapositionArrays),
    (TableComparisonArrayCollectionType.MAX_JUXTAPOSITION, MaxJuxtapositionArrays),
]

PLOT_COLLECTION_TEST_CASES = [
    (TableComparisonPlotCollectionType.PDF, PDFPlots),
    (TableComparisonPlotCollectionType.CDF, CDFPlots),
    (TableComparisonPlotCollectionType.CORRELATION_HEATMAPS, CorrelationHeatmaps),
]


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class, fake_result", SCORE_TEST_CASES)
def test_produce_score(
    mocker: MockerFixture,
    resource_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    artifact_type: TableComparisonScoreType,
    artifact_class: Type[TableComparisonArtifact[Any, Any]],
    fake_result: Any,
):
    engine = TableComparisonEngine.build(resource_spec=resource_spec)
    resources = DatasetComparisonArtifactResources(
        dataset_real=df_real, dataset_synthetic=df_synthetic
    )
    spy_get = mocker.spy(TableComparisonScoreRegistry, "get")
    spy_compute = mocker.patch.object(artifact_class, "compute", return_value=fake_result)
    result = engine.produce_score(score_type=artifact_type, resources=resources)
    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=resources)
    assert result == fake_result


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class, fake_result", SCORE_TEST_CASES)
def test_produce_dataset_comparison_score(
    mocker: MockerFixture,
    resource_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    artifact_type: TableComparisonScoreType,
    artifact_class: Type[TableComparisonArtifact[Any, Any]],
    fake_result: Any,
):
    engine = TableComparisonEngine.build(resource_spec=resource_spec)
    spy_get = mocker.spy(TableComparisonScoreRegistry, "get")
    spy_compute = mocker.patch.object(artifact_class, "compute", return_value=fake_result)
    result = engine.produce_dataset_comparison_score(
        score_type=artifact_type,
        dataset_real=df_real,
        dataset_synthetic=df_synthetic,
    )
    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=ANY)
    assert result == fake_result


@pytest.fixture
def fake_array() -> np.ndarray:
    return np.array([[1, 2], [3, 4]])


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class, fake_result", ARRAY_TEST_CASES)
def test_produce_array(
    mocker: MockerFixture,
    resource_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    artifact_type: TableComparisonArrayType,
    artifact_class: Type[TableComparisonArtifact[Any, Any]],
    fake_result: Any,
):
    engine = TableComparisonEngine.build(resource_spec=resource_spec)
    resources = DatasetComparisonArtifactResources(
        dataset_real=df_real, dataset_synthetic=df_synthetic
    )
    spy_get = mocker.spy(TableComparisonArrayRegistry, "get")
    spy_compute = mocker.patch.object(artifact_class, "compute", return_value=fake_result)
    result = engine.produce_array(array_type=artifact_type, resources=resources)
    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=resources)
    np.testing.assert_array_equal(result, fake_result)


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class, fake_result", ARRAY_TEST_CASES)
def test_produce_dataset_comparison_array(
    mocker: MockerFixture,
    resource_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    artifact_type: TableComparisonArrayType,
    artifact_class: Type[TableComparisonArtifact[Any, Any]],
    fake_result: Any,
):
    engine = TableComparisonEngine.build(resource_spec=resource_spec)
    spy_get = mocker.spy(TableComparisonArrayRegistry, "get")
    spy_compute = mocker.patch.object(artifact_class, "compute", return_value=fake_result)
    result = engine.produce_dataset_comparison_array(
        array_type=artifact_type,
        dataset_real=df_real,
        dataset_synthetic=df_synthetic,
    )
    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=ANY)
    np.testing.assert_array_equal(result, fake_result)


@pytest.fixture
def fake_figure(mocker: MockerFixture) -> Figure:
    return mocker.MagicMock(spec=Figure)


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class", PLOT_TEST_CASES)
def test_produce_plot(
    mocker: MockerFixture,
    resource_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    fake_figure: Figure,
    artifact_type: TableComparisonPlotType,
    artifact_class: Type[TableComparisonArtifact[Any, Any]],
):
    engine = TableComparisonEngine.build(resource_spec=resource_spec)
    resources = DatasetComparisonArtifactResources(
        dataset_real=df_real, dataset_synthetic=df_synthetic
    )
    spy_get = mocker.spy(TableComparisonPlotRegistry, "get")
    spy_compute = mocker.patch.object(artifact_class, "compute", return_value=fake_figure)
    result = engine.produce_plot(plot_type=artifact_type, resources=resources)
    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=resources)
    assert result is fake_figure


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class", PLOT_TEST_CASES)
def test_produce_dataset_comparison_plot(
    mocker: MockerFixture,
    resource_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    fake_figure: Figure,
    artifact_type: TableComparisonPlotType,
    artifact_class: Type[TableComparisonArtifact[Any, Any]],
):
    engine = TableComparisonEngine.build(resource_spec=resource_spec)
    spy_get = mocker.spy(TableComparisonPlotRegistry, "get")
    spy_compute = mocker.patch.object(artifact_class, "compute", return_value=fake_figure)
    result = engine.produce_dataset_comparison_plot(
        plot_type=artifact_type,
        dataset_real=df_real,
        dataset_synthetic=df_synthetic,
    )
    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=ANY)
    assert result is fake_figure


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class, fake_result", SCORE_COLLECTION_TEST_CASES)
def test_produce_score_collection(
    mocker: MockerFixture,
    resource_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    artifact_type: TableComparisonScoreCollectionType,
    artifact_class: Type[TableComparisonArtifact[Any, Any]],
    fake_result: Dict[str, float],
):
    engine = TableComparisonEngine.build(resource_spec=resource_spec)
    resources = DatasetComparisonArtifactResources(
        dataset_real=df_real, dataset_synthetic=df_synthetic
    )

    spy_get = mocker.spy(TableComparisonScoreCollectionRegistry, "get")
    spy_compute = mocker.patch.object(artifact_class, "compute", return_value=fake_result)
    result = engine.produce_score_collection(
        score_collection_type=artifact_type, resources=resources
    )
    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=resources)
    assert result == fake_result


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class, fake_result", SCORE_COLLECTION_TEST_CASES)
def test_produce_dataset_comparison_score_collection(
    mocker: MockerFixture,
    resource_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    artifact_type: TableComparisonScoreCollectionType,
    artifact_class: Type[TableComparisonArtifact[Any, Any]],
    fake_result: Dict[str, float],
):
    engine = TableComparisonEngine.build(resource_spec=resource_spec)

    spy_get = mocker.spy(TableComparisonScoreCollectionRegistry, "get")
    spy_compute = mocker.patch.object(artifact_class, "compute", return_value=fake_result)
    result = engine.produce_dataset_comparison_score_collection(
        score_collection_type=artifact_type,
        dataset_real=df_real,
        dataset_synthetic=df_synthetic,
    )
    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=ANY)
    assert result == fake_result


@pytest.fixture
def fake_array_collection() -> Dict[str, np.ndarray]:
    return {"real": np.array([1.0, 2.0]), "synthetic": np.array([1.5, 2.5])}


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class", ARRAY_COLLECTION_TEST_CASES)
def test_produce_array_collection(
    mocker: MockerFixture,
    resource_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    fake_array_collection: Dict[str, np.ndarray],
    artifact_type: TableComparisonArrayCollectionType,
    artifact_class: Type[TableComparisonArtifact[Any, Any]],
):
    engine = TableComparisonEngine.build(resource_spec=resource_spec)
    resources = DatasetComparisonArtifactResources(
        dataset_real=df_real, dataset_synthetic=df_synthetic
    )

    spy_get = mocker.spy(TableComparisonArrayCollectionRegistry, "get")
    spy_compute = mocker.patch.object(artifact_class, "compute", return_value=fake_array_collection)
    result = engine.produce_array_collection(
        array_collection_type=artifact_type, resources=resources
    )
    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=resources)
    for key in fake_array_collection:
        np.testing.assert_array_equal(result[key], fake_array_collection[key])


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class", ARRAY_COLLECTION_TEST_CASES)
def test_produce_dataset_comparison_array_collection(
    mocker: MockerFixture,
    resource_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    fake_array_collection: Dict[str, np.ndarray],
    artifact_type: TableComparisonArrayCollectionType,
    artifact_class: Type[TableComparisonArtifact[Any, Any]],
):
    engine = TableComparisonEngine.build(resource_spec=resource_spec)

    spy_get = mocker.spy(TableComparisonArrayCollectionRegistry, "get")
    spy_compute = mocker.patch.object(artifact_class, "compute", return_value=fake_array_collection)

    result = engine.produce_dataset_comparison_array_collection(
        array_collection_type=artifact_type,
        dataset_real=df_real,
        dataset_synthetic=df_synthetic,
    )

    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=ANY)
    for key in fake_array_collection:
        np.testing.assert_array_equal(result[key], fake_array_collection[key])


@pytest.fixture
def fake_plot_collection(mocker: MockerFixture) -> Dict[str, Figure]:
    return {"plot_1": mocker.MagicMock(spec=Figure), "plot_2": mocker.MagicMock(spec=Figure)}


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class", PLOT_COLLECTION_TEST_CASES)
def test_produce_plot_collection(
    mocker: MockerFixture,
    resource_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    fake_plot_collection: Dict[str, Figure],
    artifact_type: TableComparisonPlotCollectionType,
    artifact_class: Type[TableComparisonArtifact[Any, Any]],
):
    engine = TableComparisonEngine.build(resource_spec=resource_spec)
    resources = DatasetComparisonArtifactResources(
        dataset_real=df_real, dataset_synthetic=df_synthetic
    )
    spy_get = mocker.spy(TableComparisonPlotCollectionRegistry, "get")
    spy_compute = mocker.patch.object(artifact_class, "compute", return_value=fake_plot_collection)
    result = engine.produce_plot_collection(plot_collection_type=artifact_type, resources=resources)
    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=resources)
    assert result == fake_plot_collection


@pytest.mark.unit
@pytest.mark.parametrize("artifact_type, artifact_class", PLOT_COLLECTION_TEST_CASES)
def test_produce_dataset_comparison_plot_collection(
    mocker: MockerFixture,
    resource_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    fake_plot_collection: Dict[str, Figure],
    artifact_type: TableComparisonPlotCollectionType,
    artifact_class: Type[TableComparisonArtifact[Any, Any]],
):
    engine = TableComparisonEngine.build(resource_spec=resource_spec)
    spy_get = mocker.spy(TableComparisonPlotCollectionRegistry, "get")
    spy_compute = mocker.patch.object(artifact_class, "compute", return_value=fake_plot_collection)
    result = engine.produce_dataset_comparison_plot_collection(
        plot_collection_type=artifact_type,
        dataset_real=df_real,
        dataset_synthetic=df_synthetic,
    )
    spy_get.assert_called_once_with(artifact_type=artifact_type, resource_spec=resource_spec)
    spy_compute.assert_called_once_with(resources=ANY)
    assert result == fake_plot_collection
