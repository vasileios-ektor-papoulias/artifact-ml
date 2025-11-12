from typing import Callable, List, Tuple, Type

import pytest
from artifact_core.table_comparison import (
    TableComparisonArrayCollectionType,
    TableComparisonPlot,
    TableComparisonPlotCollectionType,
    TableComparisonScoreCollectionType,
    TableComparisonScoreType,
)
from artifact_core.table_comparison._artifacts.array_collections.descriptive_stats import (
    FirstQuartileJuxtapositionArrays,
    MedianJuxtapositionArrays,
    ThirdQuartileJuxtapositionArrays,
)
from artifact_core.table_comparison._artifacts.base import TableComparisonArtifact
from artifact_core.table_comparison._artifacts.plot_collections.correlations import (
    CorrelationHeatmaps,
)
from artifact_core.table_comparison._artifacts.plots.cdf import CDFPlot
from artifact_core.table_comparison._artifacts.plots.pdf import PDFPlot
from artifact_core.table_comparison._artifacts.score_collections.js import JSDistanceScores
from artifact_core.table_comparison._artifacts.scores.correlation import CorrelationDistanceScore
from artifact_core.table_comparison._artifacts.scores.mean_js import MeanJSDistanceScore
from artifact_experiment.base.callbacks.tracking import (
    ArrayCallbackHandler,
    ArrayCollectionCallbackHandler,
    PlotCallbackHandler,
    PlotCollectionCallbackHandler,
    ScoreCallbackHandler,
    ScoreCollectionCallbackHandler,
)


@pytest.fixture
def expected_score_types() -> List[TableComparisonScoreType]:
    return [
        TableComparisonScoreType.MEAN_JS_DISTANCE,
        TableComparisonScoreType.CORRELATION_DISTANCE,
    ]


@pytest.fixture
def expected_plot_types() -> List[TableComparisonPlot]:
    return [
        TableComparisonPlot.PDF,
        TableComparisonPlot.CDF,
    ]


@pytest.fixture
def expected_score_collection_types() -> List[TableComparisonScoreCollectionType]:
    return [TableComparisonScoreCollectionType.JS_DISTANCE]


@pytest.fixture
def expected_array_collection_types() -> List[TableComparisonArrayCollectionType]:
    return [
        TableComparisonArrayCollectionType.MEDIAN_JUXTAPOSITION,
        TableComparisonArrayCollectionType.FIRST_QUARTILE_JUXTAPOSITION,
        TableComparisonArrayCollectionType.THIRD_QUARTILE_JUXTAPOSITION,
    ]


@pytest.fixture
def expected_plot_collection_types() -> List[TableComparisonPlotCollectionType]:
    return [TableComparisonPlotCollectionType.CORRELATION_HEATMAPS]


@pytest.fixture
def expected_artifact_classes() -> List[Type[TableComparisonArtifact]]:
    return [
        MeanJSDistanceScore,
        CorrelationDistanceScore,
        PDFPlot,
        CDFPlot,
        JSDistanceScores,
        MedianJuxtapositionArrays,
        FirstQuartileJuxtapositionArrays,
        ThirdQuartileJuxtapositionArrays,
        CorrelationHeatmaps,
    ]


@pytest.fixture()
def handlers_factory() -> Callable[
    [],
    Tuple[
        ScoreCallbackHandler,
        ArrayCallbackHandler,
        PlotCallbackHandler,
        ScoreCollectionCallbackHandler,
        ArrayCollectionCallbackHandler,
        PlotCollectionCallbackHandler,
    ],
]:
    def factory() -> Tuple[
        ScoreCallbackHandler,
        ArrayCallbackHandler,
        PlotCallbackHandler,
        ScoreCollectionCallbackHandler,
        ArrayCollectionCallbackHandler,
        PlotCollectionCallbackHandler,
    ]:
        score_handler = ScoreCallbackHandler()
        array_handler = ArrayCallbackHandler()
        plot_handler = PlotCallbackHandler()
        score_collection_handler = ScoreCollectionCallbackHandler()
        array_collection_handler = ArrayCollectionCallbackHandler()
        plot_collection_handler = PlotCollectionCallbackHandler()

        return (
            score_handler,
            array_handler,
            plot_handler,
            score_collection_handler,
            array_collection_handler,
            plot_collection_handler,
        )

    return factory
