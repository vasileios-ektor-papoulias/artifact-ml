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
from artifact_experiment.libs.tracking.in_memory.native_run import (
    InMemoryRun,
)


@pytest.fixture
def score_logger_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRun, InMemoryRunAdapter]
    ],
) -> Callable[[Optional[str], Optional[str]], Tuple[InMemoryRunAdapter, InMemoryScoreLogger]]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[InMemoryRunAdapter, InMemoryScoreLogger]:
        _, adapter = adapter_factory(experiment_id, run_id)
        logger = InMemoryScoreLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def array_logger_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRun, InMemoryRunAdapter]
    ],
) -> Callable[[Optional[str], Optional[str]], Tuple[InMemoryRunAdapter, InMemoryArrayLogger]]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[InMemoryRunAdapter, InMemoryArrayLogger]:
        _, adapter = adapter_factory(experiment_id, run_id)
        logger = InMemoryArrayLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def plot_logger_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRun, InMemoryRunAdapter]
    ],
) -> Callable[[Optional[str], Optional[str]], Tuple[InMemoryRunAdapter, InMemoryPlotLogger]]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[InMemoryRunAdapter, InMemoryPlotLogger]:
        _, adapter = adapter_factory(experiment_id, run_id)
        logger = InMemoryPlotLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def score_collection_logger_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRun, InMemoryRunAdapter]
    ],
) -> Callable[
    [Optional[str], Optional[str]],
    Tuple[InMemoryRunAdapter, InMemoryScoreCollectionLogger],
]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[InMemoryRunAdapter, InMemoryScoreCollectionLogger]:
        _, adapter = adapter_factory(experiment_id, run_id)
        logger = InMemoryScoreCollectionLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def array_collection_logger_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRun, InMemoryRunAdapter]
    ],
) -> Callable[
    [Optional[str], Optional[str]],
    Tuple[InMemoryRunAdapter, InMemoryArrayCollectionLogger],
]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[InMemoryRunAdapter, InMemoryArrayCollectionLogger]:
        _, adapter = adapter_factory(experiment_id, run_id)
        logger = InMemoryArrayCollectionLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def plot_collection_logger_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRun, InMemoryRunAdapter]
    ],
) -> Callable[
    [Optional[str], Optional[str]],
    Tuple[InMemoryRunAdapter, InMemoryPlotCollectionLogger],
]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[InMemoryRunAdapter, InMemoryPlotCollectionLogger]:
        _, adapter = adapter_factory(experiment_id, run_id)
        logger = InMemoryPlotCollectionLogger(run=adapter)
        return adapter, logger

    return _factory
