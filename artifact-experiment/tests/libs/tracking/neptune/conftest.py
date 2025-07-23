from typing import Callable, Optional, Tuple
from unittest.mock import MagicMock

import pytest
from artifact_experiment.libs.tracking.neptune.adapter import NeptuneRunAdapter
from artifact_experiment.libs.tracking.neptune.client import NeptuneTrackingClient
from artifact_experiment.libs.tracking.neptune.loggers.array_collections import (
    NeptuneArrayCollectionLogger,
)
from artifact_experiment.libs.tracking.neptune.loggers.arrays import NeptuneArrayLogger
from artifact_experiment.libs.tracking.neptune.loggers.plot_collections import (
    NeptunePlotCollectionLogger,
)
from artifact_experiment.libs.tracking.neptune.loggers.plots import NeptunePlotLogger
from artifact_experiment.libs.tracking.neptune.loggers.score_collections import (
    NeptuneScoreCollectionLogger,
)
from artifact_experiment.libs.tracking.neptune.loggers.scores import NeptuneScoreLogger


@pytest.fixture
def native_run_factory() -> Callable[[Optional[str], Optional[str]], MagicMock]:
    def _factory(experiment_id: Optional[str] = None, run_id: Optional[str] = None) -> MagicMock:
        if experiment_id is None:
            experiment_id = "test_experiment"
        if run_id is None:
            run_id = "test_run"

        run = MagicMock()
        run.experiment_id = experiment_id
        run.run_id = run_id
        return run

    return _factory


@pytest.fixture
def adapter_factory(
    native_run_factory: Callable[[Optional[str], Optional[str]], MagicMock],
) -> Callable[[Optional[str], Optional[str]], Tuple[MagicMock, NeptuneRunAdapter]]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[MagicMock, NeptuneRunAdapter]:
        if experiment_id is None:
            experiment_id = "test_experiment"
        if run_id is None:
            run_id = "test_run"
        native_run = native_run_factory(experiment_id, run_id)
        adapter = NeptuneRunAdapter(native_run=native_run)
        return native_run, adapter

    return _factory


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


@pytest.fixture
def loggers_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[MagicMock, NeptuneRunAdapter],
    ],
) -> Callable[
    [Optional[str], Optional[str]],
    Tuple[
        NeptuneRunAdapter,
        NeptuneScoreLogger,
        NeptuneArrayLogger,
        NeptunePlotLogger,
        NeptuneScoreCollectionLogger,
        NeptuneArrayCollectionLogger,
        NeptunePlotCollectionLogger,
    ],
]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[
        NeptuneRunAdapter,
        NeptuneScoreLogger,
        NeptuneArrayLogger,
        NeptunePlotLogger,
        NeptuneScoreCollectionLogger,
        NeptuneArrayCollectionLogger,
        NeptunePlotCollectionLogger,
    ]:
        _, adapter = adapter_factory(experiment_id, run_id)
        score_logger = NeptuneScoreLogger(run=adapter)
        array_logger = NeptuneArrayLogger(run=adapter)
        plot_logger = NeptunePlotLogger(run=adapter)
        score_collection_logger = NeptuneScoreCollectionLogger(run=adapter)
        array_collection_logger = NeptuneArrayCollectionLogger(run=adapter)
        plot_collection_logger = NeptunePlotCollectionLogger(run=adapter)
        return (
            adapter,
            score_logger,
            array_logger,
            plot_logger,
            score_collection_logger,
            array_collection_logger,
            plot_collection_logger,
        )

    return _factory


@pytest.fixture
def client_factory(
    loggers_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[
            NeptuneRunAdapter,
            NeptuneScoreLogger,
            NeptuneArrayLogger,
            NeptunePlotLogger,
            NeptuneScoreCollectionLogger,
            NeptuneArrayCollectionLogger,
            NeptunePlotCollectionLogger,
        ],
    ],
) -> Callable[
    [Optional[str], Optional[str]],
    Tuple[
        NeptuneRunAdapter,
        NeptuneScoreLogger,
        NeptuneArrayLogger,
        NeptunePlotLogger,
        NeptuneScoreCollectionLogger,
        NeptuneArrayCollectionLogger,
        NeptunePlotCollectionLogger,
        NeptuneTrackingClient,
    ],
]:
    def _factory(
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[
        NeptuneRunAdapter,
        NeptuneScoreLogger,
        NeptuneArrayLogger,
        NeptunePlotLogger,
        NeptuneScoreCollectionLogger,
        NeptuneArrayCollectionLogger,
        NeptunePlotCollectionLogger,
        NeptuneTrackingClient,
    ]:
        (
            adapter,
            score_logger,
            array_logger,
            plot_logger,
            score_collection_logger,
            array_collection_logger,
            plot_collection_logger,
        ) = loggers_factory(experiment_id, run_id)
        client = NeptuneTrackingClient(
            run=adapter,
            score_logger=score_logger,
            array_logger=array_logger,
            plot_logger=plot_logger,
            score_collection_logger=score_collection_logger,
            array_collection_logger=array_collection_logger,
            plot_collection_logger=plot_collection_logger,
        )
        return (
            adapter,
            score_logger,
            array_logger,
            plot_logger,
            score_collection_logger,
            array_collection_logger,
            plot_collection_logger,
            client,
        )

    return _factory
