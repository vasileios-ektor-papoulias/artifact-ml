from typing import Callable, Optional, Tuple

import matplotlib
import pytest

from tests.base.tracking.dummy.adapter import DummyNativeRun, DummyRunAdapter
from tests.base.tracking.dummy.client import DummyTrackingClient
from tests.base.tracking.dummy.logger import DummyArtifactLogger

matplotlib.use("Agg")


@pytest.fixture
def native_run_factory() -> Callable[[Optional[str], Optional[str]], DummyNativeRun]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> DummyNativeRun:
        if experiment_id is None:
            experiment_id = "test_experiment"
        if run_id is None:
            run_id = "test_run"

        run = DummyNativeRun(experiment_id=experiment_id, run_id=run_id)
        return run

    return _factory


@pytest.fixture
def adapter_factory() -> Callable[
    [Optional[str], Optional[str]], Tuple[DummyNativeRun, DummyRunAdapter]
]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[DummyNativeRun, DummyRunAdapter]:
        if experiment_id is None:
            experiment_id = "test_experiment"
        if run_id is None:
            run_id = "test_run"
        native_run = DummyNativeRun(experiment_id=experiment_id, run_id=run_id)
        adapter = DummyRunAdapter.from_native_run(native_run=native_run)
        return native_run, adapter

    return _factory


@pytest.fixture
def logger_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[DummyNativeRun, DummyRunAdapter]
    ],
) -> Callable[[Optional[str], Optional[str]], Tuple[DummyRunAdapter, DummyArtifactLogger]]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[DummyRunAdapter, DummyArtifactLogger]:
        _, adapter = adapter_factory(experiment_id, run_id)
        logger = DummyArtifactLogger(run=adapter)
        return adapter, logger

    return _factory


@pytest.fixture
def client_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[DummyNativeRun, DummyRunAdapter]
    ],
    logger_factory: Callable[
        [Optional[DummyRunAdapter]], Tuple[DummyRunAdapter, DummyArtifactLogger]
    ],
) -> Callable[
    [Optional[str], Optional[str]], Tuple[DummyRunAdapter, DummyArtifactLogger, DummyTrackingClient]
]:
    def _factory(
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[DummyRunAdapter, DummyArtifactLogger, DummyTrackingClient]:
        _, adapter = adapter_factory(experiment_id, run_id)
        adapter, logger = logger_factory(adapter)
        client = DummyTrackingClient(
            run=adapter,
            score_logger=logger,
            array_logger=logger,
            plot_logger=logger,
            score_collection_logger=logger,
            array_collection_logger=logger,
            plot_collection_logger=logger,
        )
        return adapter, logger, client

    return _factory
