from typing import Callable, Optional, Tuple

import pytest
from artifact_experiment.libs.tracking.filesystem.adapter import FilesystemRunAdapter
from artifact_experiment.libs.tracking.filesystem.loggers.array_collections import (
    FilesystemArrayCollectionLogger,
)
from artifact_experiment.libs.tracking.filesystem.loggers.arrays import FilesystemArrayLogger
from artifact_experiment.libs.tracking.filesystem.loggers.plot_collections import (
    FilesystemPlotCollectionLogger,
)
from artifact_experiment.libs.tracking.filesystem.loggers.plots import FilesystemPlotLogger
from artifact_experiment.libs.tracking.filesystem.loggers.score_collections import (
    FilesystemScoreCollectionLogger,
)
from artifact_experiment.libs.tracking.filesystem.loggers.scores import FilesystemScoreLogger
from artifact_experiment.libs.tracking.filesystem.native_run import FilesystemRun


@pytest.fixture
def score_logger_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[FilesystemRun, FilesystemRunAdapter]
    ],
) -> Callable[[Optional[str], Optional[str]], Tuple[FilesystemRunAdapter, FilesystemScoreLogger]]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[FilesystemRunAdapter, FilesystemScoreLogger]:
        _, adapter = adapter_factory(experiment_id, run_id)
        logger = FilesystemScoreLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def array_logger_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[FilesystemRun, FilesystemRunAdapter]
    ],
) -> Callable[[Optional[str], Optional[str]], Tuple[FilesystemRunAdapter, FilesystemArrayLogger]]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[FilesystemRunAdapter, FilesystemArrayLogger]:
        _, adapter = adapter_factory(experiment_id, run_id)
        logger = FilesystemArrayLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def plot_logger_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[FilesystemRun, FilesystemRunAdapter]
    ],
) -> Callable[[Optional[str], Optional[str]], Tuple[FilesystemRunAdapter, FilesystemPlotLogger]]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[FilesystemRunAdapter, FilesystemPlotLogger]:
        _, adapter = adapter_factory(experiment_id, run_id)
        logger = FilesystemPlotLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def score_collection_logger_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[FilesystemRun, FilesystemRunAdapter]
    ],
) -> Callable[
    [Optional[str], Optional[str]], Tuple[FilesystemRunAdapter, FilesystemScoreCollectionLogger]
]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[FilesystemRunAdapter, FilesystemScoreCollectionLogger]:
        _, adapter = adapter_factory(experiment_id, run_id)
        logger = FilesystemScoreCollectionLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def array_collection_logger_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[FilesystemRun, FilesystemRunAdapter]
    ],
) -> Callable[
    [Optional[str], Optional[str]], Tuple[FilesystemRunAdapter, FilesystemArrayCollectionLogger]
]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[FilesystemRunAdapter, FilesystemArrayCollectionLogger]:
        _, adapter = adapter_factory(experiment_id, run_id)
        logger = FilesystemArrayCollectionLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def plot_collection_logger_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[FilesystemRun, FilesystemRunAdapter]
    ],
) -> Callable[
    [Optional[str], Optional[str]], Tuple[FilesystemRunAdapter, FilesystemPlotCollectionLogger]
]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[FilesystemRunAdapter, FilesystemPlotCollectionLogger]:
        _, adapter = adapter_factory(experiment_id, run_id)
        logger = FilesystemPlotCollectionLogger(run=adapter)
        return adapter, logger

    return _factory
