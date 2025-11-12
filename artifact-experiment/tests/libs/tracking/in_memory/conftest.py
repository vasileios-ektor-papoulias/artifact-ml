from typing import Callable, Optional, Tuple

import pytest
from artifact_experiment.libs.tracking.in_memory.adapter import (
    InMemoryRunAdapter,
)
from artifact_experiment.libs.tracking.in_memory.client import (
    InMemoryTrackingClient,
)
from artifact_experiment.libs.tracking.in_memory.native_run import (
    InMemoryRun,
)


@pytest.fixture
def native_run_factory() -> Callable[[Optional[str], Optional[str]], InMemoryRun]:
    def _factory(experiment_id: Optional[str] = None, run_id: Optional[str] = None) -> InMemoryRun:
        if experiment_id is None:
            experiment_id = "default_experiment_id"
        if run_id is None:
            run_id = "default_run_id"
        native_run = InMemoryRun(experiment_id=experiment_id, run_id=run_id)
        return native_run

    return _factory


@pytest.fixture
def adapter_factory(
    native_run_factory: Callable[[Optional[str], Optional[str]], InMemoryRun],
) -> Callable[[Optional[str], Optional[str]], Tuple[InMemoryRun, InMemoryRunAdapter]]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[InMemoryRun, InMemoryRunAdapter]:
        if experiment_id is None:
            experiment_id = "default_experiment_id"
        if run_id is None:
            run_id = "default_run_id"
        native_run = native_run_factory(experiment_id, run_id)
        adapter = InMemoryRunAdapter(native_run=native_run)
        return native_run, adapter

    return _factory


@pytest.fixture
def populated_adapter_factory(
    request,
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRun, InMemoryRunAdapter]
    ],
) -> Callable[[Optional[str], Optional[str]], Tuple[InMemoryRun, InMemoryRunAdapter]]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[InMemoryRun, InMemoryRunAdapter]:
        native_run, adapter = adapter_factory(experiment_id, run_id)
        fixture_names = request.param
        score_idx = 1
        array_idx = 1
        plot_idx = 1
        score_collection_idx = 1
        array_collection_idx = 1
        plot_collection_idx = 1
        for fixture_name in fixture_names:
            artifact = request.getfixturevalue(fixture_name)
            if isinstance(artifact, float):
                native_run.log_score(key=f"test_score/{score_idx}", score=artifact)
                score_idx += 1
            elif isinstance(artifact, Array):
                native_run.log_array(key=f"test_array/{array_idx}", array=artifact)
                array_idx += 1
            elif hasattr(artifact, "add_subplot"):
                native_run.log_plot(key=f"test_plot/{plot_idx}", plot=artifact)
                plot_idx += 1
            elif isinstance(artifact, dict):
                values = artifact.values()
                if all(isinstance(v, float) for v in values):
                    native_run.log_score_collection(
                        key=f"test_score_collection/{score_collection_idx}",
                        score_collection=artifact,
                    )
                    score_collection_idx += 1
                elif all(isinstance(v, Array) for v in values):
                    native_run.log_array_collection(
                        key=f"test_array_collection/{array_collection_idx}",
                        array_collection=artifact,
                    )
                    array_collection_idx += 1
                elif all(hasattr(v, "add_subplot") for v in values):
                    native_run.log_plot_collection(
                        key=f"test_plot_collection/{plot_collection_idx}",
                        plot_collection=artifact,
                    )
                    plot_collection_idx += 1
        return native_run, adapter

    return _factory


@pytest.fixture
def client_factory(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRun, InMemoryRunAdapter]
    ],
) -> Callable[[Optional[str], Optional[str]], Tuple[InMemoryRunAdapter, InMemoryTrackingClient]]:
    def _factory(
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[InMemoryRunAdapter, InMemoryTrackingClient]:
        _, adapter = adapter_factory(experiment_id, run_id)
        client = InMemoryTrackingClient.from_run(run=adapter)
        return adapter, client

    return _factory
