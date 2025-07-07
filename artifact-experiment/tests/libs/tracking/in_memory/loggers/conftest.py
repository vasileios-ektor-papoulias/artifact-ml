from typing import Callable, Optional
from unittest.mock import Mock

import pytest
from artifact_experiment.libs.tracking.in_memory.adapter import (
    InMemoryTrackingAdapter,
)
from artifact_experiment.libs.tracking.in_memory.loggers.array_collections import (
    InMemoryArrayCollectionLogger,
)
from artifact_experiment.libs.tracking.in_memory.loggers.arrays import (
    InMemoryArrayLogger,
)
from artifact_experiment.libs.tracking.in_memory.loggers.plot_collections import (
    InMemoryPlotCollectionLogger,
)
from artifact_experiment.libs.tracking.in_memory.loggers.plots import (
    InMemoryPlotLogger,
)
from artifact_experiment.libs.tracking.in_memory.loggers.score_collections import (
    InMemoryScoreCollectionLogger,
)
from artifact_experiment.libs.tracking.in_memory.loggers.scores import (
    InMemoryScoreLogger,
)


@pytest.fixture
def mock_adapter() -> InMemoryTrackingAdapter:
    adapter = Mock(spec=InMemoryTrackingAdapter)
    adapter.experiment_id = "test_experiment"
    adapter.run_id = "test_run"
    adapter.n_scores = 0
    adapter.n_arrays = 0
    adapter.n_plots = 0
    adapter.n_score_collections = 0
    adapter.n_array_collections = 0
    adapter.n_plot_collections = 0
    mock_native_run = Mock()
    mock_native_run.dict_scores = {}
    mock_native_run.dict_arrays = {}
    mock_native_run.dict_plots = {}
    mock_native_run.dict_score_collections = {}
    mock_native_run.dict_array_collections = {}
    mock_native_run.dict_plot_collections = {}
    adapter._native_run = mock_native_run
    return adapter


@pytest.fixture
def score_logger_factory(
    mock_adapter: InMemoryTrackingAdapter,
) -> Callable[[Optional[InMemoryTrackingAdapter]], InMemoryScoreLogger]:
    def _factory(
        adapter: Optional[InMemoryTrackingAdapter] = None,
    ) -> InMemoryScoreLogger:
        if adapter is None:
            adapter = mock_adapter
        return InMemoryScoreLogger(run=adapter)

    return _factory


@pytest.fixture
def array_logger_factory(
    mock_adapter: InMemoryTrackingAdapter,
) -> Callable[[Optional[InMemoryTrackingAdapter]], InMemoryArrayLogger]:
    def _factory(
        adapter: Optional[InMemoryTrackingAdapter] = None,
    ) -> InMemoryArrayLogger:
        if adapter is None:
            adapter = mock_adapter
        return InMemoryArrayLogger(run=adapter)

    return _factory


@pytest.fixture
def plot_logger_factory(
    mock_adapter: InMemoryTrackingAdapter,
) -> Callable[[Optional[InMemoryTrackingAdapter]], InMemoryPlotLogger]:
    def _factory(
        adapter: Optional[InMemoryTrackingAdapter] = None,
    ) -> InMemoryPlotLogger:
        if adapter is None:
            adapter = mock_adapter
        return InMemoryPlotLogger(run=adapter)

    return _factory


@pytest.fixture
def score_collection_logger_factory(
    mock_adapter: InMemoryTrackingAdapter,
) -> Callable[[Optional[InMemoryTrackingAdapter]], InMemoryScoreCollectionLogger]:
    def _factory(
        adapter: Optional[InMemoryTrackingAdapter] = None,
    ) -> InMemoryScoreCollectionLogger:
        if adapter is None:
            adapter = mock_adapter
        return InMemoryScoreCollectionLogger(run=adapter)

    return _factory


@pytest.fixture
def array_collection_logger_factory(
    mock_adapter: InMemoryTrackingAdapter,
) -> Callable[[Optional[InMemoryTrackingAdapter]], InMemoryArrayCollectionLogger]:
    def _factory(
        adapter: Optional[InMemoryTrackingAdapter] = None,
    ) -> InMemoryArrayCollectionLogger:
        if adapter is None:
            adapter = mock_adapter
        return InMemoryArrayCollectionLogger(run=adapter)

    return _factory


@pytest.fixture
def plot_collection_logger_factory(
    mock_adapter: InMemoryTrackingAdapter,
) -> Callable[[Optional[InMemoryTrackingAdapter]], InMemoryPlotCollectionLogger]:
    def _factory(
        adapter: Optional[InMemoryTrackingAdapter] = None,
    ) -> InMemoryPlotCollectionLogger:
        if adapter is None:
            adapter = mock_adapter
        return InMemoryPlotCollectionLogger(run=adapter)

    return _factory
