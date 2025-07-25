import pytest
from artifact_core.table_comparison import (
    TableComparisonArrayCollectionType,
    TableComparisonArrayType,
    TableComparisonPlotCollectionType,
    TableComparisonPlotType,
    TableComparisonScoreCollectionType,
    TableComparisonScoreType,
    TabularDataSpec,
)
from artifact_core.table_comparison.registries.array_collections.registry import (
    TableComparisonArrayCollectionRegistry,
)
from artifact_core.table_comparison.registries.arrays.registry import TableComparisonArrayRegistry
from artifact_core.table_comparison.registries.plot_collections.registry import (
    TableComparisonPlotCollectionRegistry,
)
from artifact_core.table_comparison.registries.plots.registry import TableComparisonPlotRegistry
from artifact_core.table_comparison.registries.score_collections.registry import (
    TableComparisonScoreCollectionRegistry,
)
from artifact_core.table_comparison.registries.scores.registry import TableComparisonScoreRegistry
from artifact_experiment.base.callbacks.artifact import (
    ArtifactArrayCallback,
    ArtifactArrayCollectionCallback,
    ArtifactPlotCallback,
    ArtifactPlotCollectionCallback,
    ArtifactScoreCallback,
    ArtifactScoreCollectionCallback,
)
from artifact_experiment.table_comparison.callback_factory import TableComparisonCallbackFactory
from pytest_mock import MockerFixture


@pytest.mark.parametrize(
    "score_type",
    [TableComparisonScoreType.MEAN_JS_DISTANCE, TableComparisonScoreType.CORRELATION_DISTANCE],
)
def test_build_score_callback(
    mocker: MockerFixture, tabular_data_spec: TabularDataSpec, score_type: TableComparisonScoreType
):
    spy_registry_get = mocker.spy(TableComparisonScoreRegistry, "get")
    callback = TableComparisonCallbackFactory.build_score_callback(
        score_type=score_type, resource_spec=tabular_data_spec
    )
    spy_registry_get.assert_called_once_with(
        artifact_type=score_type, resource_spec=tabular_data_spec
    )
    assert isinstance(callback, ArtifactScoreCallback)
    assert callback.key == score_type.name


@pytest.mark.parametrize("array_type", [])
def test_build_array_callback(
    mocker: MockerFixture, tabular_data_spec: TabularDataSpec, array_type: TableComparisonArrayType
):
    spy_registry_get = mocker.spy(TableComparisonArrayRegistry, "get")
    callback = TableComparisonCallbackFactory.build_array_callback(
        array_type=array_type, resource_spec=tabular_data_spec
    )
    spy_registry_get.assert_called_once_with(
        artifact_type=array_type, resource_spec=tabular_data_spec
    )
    assert isinstance(callback, ArtifactArrayCallback)
    assert callback.key == array_type.name


@pytest.mark.parametrize(
    "plot_type",
    [
        TableComparisonPlotType.PDF,
        TableComparisonPlotType.CDF,
        TableComparisonPlotType.DESCRIPTIVE_STATS_ALIGNMENT,
        TableComparisonPlotType.MEAN_ALIGNMENT,
        TableComparisonPlotType.STD_ALIGNMENT,
        TableComparisonPlotType.VARIANCE_ALIGNMENT,
        TableComparisonPlotType.MEDIAN_ALIGNMENT,
        TableComparisonPlotType.FIRST_QUARTILE_ALIGNMENT,
        TableComparisonPlotType.THIRD_QUARTILE_ALIGNMENT,
        TableComparisonPlotType.MIN_ALIGNMENT,
        TableComparisonPlotType.MAX_ALIGNMENT,
        TableComparisonPlotType.CORRELATION_HEATMAP_JUXTAPOSITION,
        TableComparisonPlotType.PCA_JUXTAPOSITION,
        TableComparisonPlotType.TRUNCATED_SVD_JUXTAPOSITION,
        TableComparisonPlotType.TSNE_JUXTAPOSITION,
    ],
)
def test_build_plot_callback(
    mocker: MockerFixture, tabular_data_spec: TabularDataSpec, plot_type: TableComparisonPlotType
):
    spy_registry_get = mocker.spy(TableComparisonPlotRegistry, "get")
    callback = TableComparisonCallbackFactory.build_plot_callback(
        plot_type=plot_type, resource_spec=tabular_data_spec
    )
    spy_registry_get.assert_called_once_with(
        artifact_type=plot_type, resource_spec=tabular_data_spec
    )
    assert isinstance(callback, ArtifactPlotCallback)
    assert callback.key == plot_type.name


@pytest.mark.parametrize("score_collection_type", [TableComparisonScoreCollectionType.JS_DISTANCE])
def test_build_score_collection_callback(
    mocker: MockerFixture,
    tabular_data_spec: TabularDataSpec,
    score_collection_type: TableComparisonScoreCollectionType,
):
    spy_registry_get = mocker.spy(TableComparisonScoreCollectionRegistry, "get")
    callback = TableComparisonCallbackFactory.build_score_collection_callback(
        score_collection_type=score_collection_type, resource_spec=tabular_data_spec
    )
    spy_registry_get.assert_called_once_with(
        artifact_type=score_collection_type, resource_spec=tabular_data_spec
    )
    assert isinstance(callback, ArtifactScoreCollectionCallback)
    assert callback.key == score_collection_type.name


@pytest.mark.parametrize(
    "array_collection_type",
    [
        TableComparisonArrayCollectionType.MEAN_JUXTAPOSITION,
        TableComparisonArrayCollectionType.STD_JUXTAPOSITION,
        TableComparisonArrayCollectionType.VARIANCE_JUXTAPOSITION,
        TableComparisonArrayCollectionType.MEDIAN_JUXTAPOSITION,
        TableComparisonArrayCollectionType.FIRST_QUARTILE_JUXTAPOSITION,
        TableComparisonArrayCollectionType.THIRD_QUARTILE_JUXTAPOSITION,
        TableComparisonArrayCollectionType.MIN_JUXTAPOSITION,
        TableComparisonArrayCollectionType.MAX_JUXTAPOSITION,
    ],
)
def test_build_array_collection_callback(
    mocker: MockerFixture,
    tabular_data_spec: TabularDataSpec,
    array_collection_type: TableComparisonArrayCollectionType,
):
    spy_registry_get = mocker.spy(TableComparisonArrayCollectionRegistry, "get")
    callback = TableComparisonCallbackFactory.build_array_collection_callback(
        array_collection_type=array_collection_type, resource_spec=tabular_data_spec
    )
    spy_registry_get.assert_called_once_with(
        artifact_type=array_collection_type, resource_spec=tabular_data_spec
    )
    assert isinstance(callback, ArtifactArrayCollectionCallback)
    assert callback.key == array_collection_type.name


@pytest.mark.parametrize(
    "plot_collection_type",
    [
        TableComparisonPlotCollectionType.PDF,
        TableComparisonPlotCollectionType.CDF,
        TableComparisonPlotCollectionType.CORRELATION_HEATMAPS,
    ],
)
def test_build_plot_collection_callback(
    mocker: MockerFixture,
    tabular_data_spec: TabularDataSpec,
    plot_collection_type: TableComparisonPlotCollectionType,
):
    spy_registry_get = mocker.spy(TableComparisonPlotCollectionRegistry, "get")
    callback = TableComparisonCallbackFactory.build_plot_collection_callback(
        plot_collection_type=plot_collection_type, resource_spec=tabular_data_spec
    )
    spy_registry_get.assert_called_once_with(
        artifact_type=plot_collection_type, resource_spec=tabular_data_spec
    )
    assert isinstance(callback, ArtifactPlotCollectionCallback)
    assert callback.key == plot_collection_type.name
