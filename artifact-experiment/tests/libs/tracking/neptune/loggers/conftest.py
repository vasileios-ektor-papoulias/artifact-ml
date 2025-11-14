from typing import Callable, Optional, Tuple
from unittest.mock import MagicMock

import pytest
from artifact_experiment._impl.neptune.adapter import NeptuneRunAdapter
from artifact_experiment._impl.neptune.loggers.array_collections import (
    NeptuneArrayCollectionLogger,
)
from artifact_experiment._impl.neptune.loggers.arrays import NeptuneArrayLogger
from artifact_experiment._impl.neptune.loggers.plot_collections import (
    NeptunePlotCollectionLogger,
)
from artifact_experiment._impl.neptune.loggers.plots import NeptunePlotLogger
from artifact_experiment._impl.neptune.loggers.score_collections import (
    NeptuneScoreCollectionLogger,
)
from artifact_experiment._impl.neptune.loggers.scores import NeptuneScoreLogger


@pytest.fixture
def score_logger_factory(
    adapter_factory: Callable[[Optional[str], Optional[str]], Tuple[MagicMock, NeptuneRunAdapter]],
) -> Callable[[Optional[str], Optional[str]], Tuple[NeptuneRunAdapter, NeptuneScoreLogger]]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[NeptuneRunAdapter, NeptuneScoreLogger]:
        _, adapter = adapter_factory(experiment_id, run_id)
        logger = NeptuneScoreLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def array_logger_factory(
    adapter_factory: Callable[[Optional[str], Optional[str]], Tuple[MagicMock, NeptuneRunAdapter]],
) -> Callable[[Optional[str], Optional[str]], Tuple[NeptuneRunAdapter, NeptuneArrayLogger]]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[NeptuneRunAdapter, NeptuneArrayLogger]:
        _, adapter = adapter_factory(experiment_id, run_id)
        logger = NeptuneArrayLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def plot_logger_factory(
    adapter_factory: Callable[[Optional[str], Optional[str]], Tuple[MagicMock, NeptuneRunAdapter]],
) -> Callable[[Optional[str], Optional[str]], Tuple[NeptuneRunAdapter, NeptunePlotLogger]]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[NeptuneRunAdapter, NeptunePlotLogger]:
        _, adapter = adapter_factory(experiment_id, run_id)
        logger = NeptunePlotLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def score_collection_logger_factory(
    adapter_factory: Callable[[Optional[str], Optional[str]], Tuple[MagicMock, NeptuneRunAdapter]],
) -> Callable[
    [Optional[str], Optional[str]], Tuple[NeptuneRunAdapter, NeptuneScoreCollectionLogger]
]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[NeptuneRunAdapter, NeptuneScoreCollectionLogger]:
        _, adapter = adapter_factory(experiment_id, run_id)
        logger = NeptuneScoreCollectionLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def array_collection_logger_factory(
    adapter_factory: Callable[[Optional[str], Optional[str]], Tuple[MagicMock, NeptuneRunAdapter]],
) -> Callable[
    [Optional[str], Optional[str]], Tuple[NeptuneRunAdapter, NeptuneArrayCollectionLogger]
]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[NeptuneRunAdapter, NeptuneArrayCollectionLogger]:
        _, adapter = adapter_factory(experiment_id, run_id)
        logger = NeptuneArrayCollectionLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def plot_collection_logger_factory(
    adapter_factory: Callable[[Optional[str], Optional[str]], Tuple[MagicMock, NeptuneRunAdapter]],
) -> Callable[
    [Optional[str], Optional[str]], Tuple[NeptuneRunAdapter, NeptunePlotCollectionLogger]
]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[NeptuneRunAdapter, NeptunePlotCollectionLogger]:
        _, adapter = adapter_factory(experiment_id, run_id)
        logger = NeptunePlotCollectionLogger(run=adapter)
        return adapter, logger

    return _factory
