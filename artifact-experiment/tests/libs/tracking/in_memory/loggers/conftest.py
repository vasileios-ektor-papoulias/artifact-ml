from typing import Callable, Optional, Tuple

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
def score_logger_factory(
    adapter_factory: Callable[[Optional[str], Optional[str]], InMemoryTrackingAdapter],
) -> Callable[
    [Optional[InMemoryTrackingAdapter]], Tuple[InMemoryTrackingAdapter, InMemoryScoreLogger]
]:
    def _factory(
        adapter: Optional[InMemoryTrackingAdapter] = None,
    ) -> Tuple[InMemoryTrackingAdapter, InMemoryScoreLogger]:
        if adapter is None:
            adapter = adapter_factory(None, None)
        logger = InMemoryScoreLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def array_logger_factory(
    adapter_factory: Callable[[Optional[str], Optional[str]], InMemoryTrackingAdapter],
) -> Callable[
    [Optional[InMemoryTrackingAdapter]], Tuple[InMemoryTrackingAdapter, InMemoryArrayLogger]
]:
    def _factory(
        adapter: Optional[InMemoryTrackingAdapter] = None,
    ) -> Tuple[InMemoryTrackingAdapter, InMemoryArrayLogger]:
        if adapter is None:
            adapter = adapter_factory(None, None)
        logger = InMemoryArrayLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def plot_logger_factory(
    adapter_factory: Callable[[Optional[str], Optional[str]], InMemoryTrackingAdapter],
) -> Callable[
    [Optional[InMemoryTrackingAdapter]], Tuple[InMemoryTrackingAdapter, InMemoryPlotLogger]
]:
    def _factory(
        adapter: Optional[InMemoryTrackingAdapter] = None,
    ) -> Tuple[InMemoryTrackingAdapter, InMemoryPlotLogger]:
        if adapter is None:
            adapter = adapter_factory(None, None)
        logger = InMemoryPlotLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def score_collection_logger_factory(
    adapter_factory: Callable[[Optional[str], Optional[str]], InMemoryTrackingAdapter],
) -> Callable[
    [Optional[InMemoryTrackingAdapter]],
    Tuple[InMemoryTrackingAdapter, InMemoryScoreCollectionLogger],
]:
    def _factory(
        adapter: Optional[InMemoryTrackingAdapter] = None,
    ) -> Tuple[InMemoryTrackingAdapter, InMemoryScoreCollectionLogger]:
        if adapter is None:
            adapter = adapter_factory(None, None)
        logger = InMemoryScoreCollectionLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def array_collection_logger_factory(
    adapter_factory: Callable[[Optional[str], Optional[str]], InMemoryTrackingAdapter],
) -> Callable[
    [Optional[InMemoryTrackingAdapter]],
    Tuple[InMemoryTrackingAdapter, InMemoryArrayCollectionLogger],
]:
    def _factory(
        adapter: Optional[InMemoryTrackingAdapter] = None,
    ) -> Tuple[InMemoryTrackingAdapter, InMemoryArrayCollectionLogger]:
        if adapter is None:
            adapter = adapter_factory(None, None)
        logger = InMemoryArrayCollectionLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def plot_collection_logger_factory(
    adapter_factory: Callable[[Optional[str], Optional[str]], InMemoryTrackingAdapter],
) -> Callable[
    [Optional[InMemoryTrackingAdapter]],
    Tuple[InMemoryTrackingAdapter, InMemoryPlotCollectionLogger],
]:
    def _factory(
        adapter: Optional[InMemoryTrackingAdapter] = None,
    ) -> Tuple[InMemoryTrackingAdapter, InMemoryPlotCollectionLogger]:
        if adapter is None:
            adapter = adapter_factory(None, None)
        logger = InMemoryPlotCollectionLogger(run=adapter)
        return adapter, logger

    return _factory
