from typing import Callable, Optional, Tuple

import pytest
from artifact_experiment.libs.tracking.in_memory.adapter import (
    InMemoryRunAdapter,
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
    adapter_factory: Callable[[Optional[str], Optional[str]], InMemoryRunAdapter],
) -> Callable[[Optional[InMemoryRunAdapter]], Tuple[InMemoryRunAdapter, InMemoryScoreLogger]]:
    def _factory(
        adapter: Optional[InMemoryRunAdapter] = None,
    ) -> Tuple[InMemoryRunAdapter, InMemoryScoreLogger]:
        if adapter is None:
            adapter = adapter_factory(None, None)
        logger = InMemoryScoreLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def array_logger_factory(
    adapter_factory: Callable[[Optional[str], Optional[str]], InMemoryRunAdapter],
) -> Callable[[Optional[InMemoryRunAdapter]], Tuple[InMemoryRunAdapter, InMemoryArrayLogger]]:
    def _factory(
        adapter: Optional[InMemoryRunAdapter] = None,
    ) -> Tuple[InMemoryRunAdapter, InMemoryArrayLogger]:
        if adapter is None:
            adapter = adapter_factory(None, None)
        logger = InMemoryArrayLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def plot_logger_factory(
    adapter_factory: Callable[[Optional[str], Optional[str]], InMemoryRunAdapter],
) -> Callable[[Optional[InMemoryRunAdapter]], Tuple[InMemoryRunAdapter, InMemoryPlotLogger]]:
    def _factory(
        adapter: Optional[InMemoryRunAdapter] = None,
    ) -> Tuple[InMemoryRunAdapter, InMemoryPlotLogger]:
        if adapter is None:
            adapter = adapter_factory(None, None)
        logger = InMemoryPlotLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def score_collection_logger_factory(
    adapter_factory: Callable[[Optional[str], Optional[str]], InMemoryRunAdapter],
) -> Callable[
    [Optional[InMemoryRunAdapter]],
    Tuple[InMemoryRunAdapter, InMemoryScoreCollectionLogger],
]:
    def _factory(
        adapter: Optional[InMemoryRunAdapter] = None,
    ) -> Tuple[InMemoryRunAdapter, InMemoryScoreCollectionLogger]:
        if adapter is None:
            adapter = adapter_factory(None, None)
        logger = InMemoryScoreCollectionLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def array_collection_logger_factory(
    adapter_factory: Callable[[Optional[str], Optional[str]], InMemoryRunAdapter],
) -> Callable[
    [Optional[InMemoryRunAdapter]],
    Tuple[InMemoryRunAdapter, InMemoryArrayCollectionLogger],
]:
    def _factory(
        adapter: Optional[InMemoryRunAdapter] = None,
    ) -> Tuple[InMemoryRunAdapter, InMemoryArrayCollectionLogger]:
        if adapter is None:
            adapter = adapter_factory(None, None)
        logger = InMemoryArrayCollectionLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def plot_collection_logger_factory(
    adapter_factory: Callable[[Optional[str], Optional[str]], InMemoryRunAdapter],
) -> Callable[
    [Optional[InMemoryRunAdapter]],
    Tuple[InMemoryRunAdapter, InMemoryPlotCollectionLogger],
]:
    def _factory(
        adapter: Optional[InMemoryRunAdapter] = None,
    ) -> Tuple[InMemoryRunAdapter, InMemoryPlotCollectionLogger]:
        if adapter is None:
            adapter = adapter_factory(None, None)
        logger = InMemoryPlotCollectionLogger(run=adapter)
        return adapter, logger

    return _factory
