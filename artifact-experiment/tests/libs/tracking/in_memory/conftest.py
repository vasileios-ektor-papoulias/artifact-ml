from typing import Callable, Optional

import matplotlib
import numpy as np
import pytest
from artifact_experiment.libs.tracking.in_memory.adapter import (
    InMemoryTrackingAdapter,
)
from artifact_experiment.libs.tracking.in_memory.client import (
    InMemoryTrackingClient,
)
from artifact_experiment.libs.tracking.in_memory.native_run import (
    InMemoryNativeRun,
)

matplotlib.use("Agg")


@pytest.fixture
def native_run_factory() -> Callable[[Optional[str], Optional[str]], InMemoryNativeRun]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> InMemoryNativeRun:
        if experiment_id is None:
            experiment_id = "test_experiment"
        if run_id is None:
            run_id = "test_run"
        return InMemoryNativeRun(experiment_id=experiment_id, run_id=run_id)

    return _factory


@pytest.fixture
def adapter_factory() -> Callable[[Optional[str], Optional[str]], InMemoryTrackingAdapter]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> InMemoryTrackingAdapter:
        if experiment_id is None:
            experiment_id = "test_experiment"
        if run_id is None:
            run_id = "test_run"
        return InMemoryTrackingAdapter.build(experiment_id=experiment_id, run_id=run_id)

    return _factory


@pytest.fixture
def populated_adapter(
    request,
    adapter_factory: Callable[[Optional[str], Optional[str]], InMemoryTrackingAdapter],
) -> InMemoryTrackingAdapter:
    adapter = adapter_factory(None, None)
    fixture_names = request.param

    with adapter.native() as native_run:
        score_idx = array_idx = plot_idx = collection_idx = 1

        for fixture_name in fixture_names:
            artifact = request.getfixturevalue(fixture_name)

            if isinstance(artifact, float):
                native_run.dict_scores[f"test_score/{score_idx}"] = artifact
                score_idx += 1
            elif isinstance(artifact, np.ndarray):
                native_run.dict_arrays[f"test_array/{array_idx}"] = artifact
                array_idx += 1
            elif hasattr(artifact, "add_subplot"):
                native_run.dict_plots[f"test_plot/{plot_idx}"] = artifact
                plot_idx += 1
            elif isinstance(artifact, dict):
                values = artifact.values()
                if all(isinstance(v, float) for v in values):
                    collections = native_run.dict_score_collections
                    collections[f"test_collection/{collection_idx}"] = artifact
                    collection_idx += 1
                elif all(isinstance(v, np.ndarray) for v in values):
                    collections = native_run.dict_array_collections
                    collections[f"test_collection/{collection_idx}"] = artifact
                    collection_idx += 1
                elif all(hasattr(v, "add_subplot") for v in values):
                    collections = native_run.dict_plot_collections
                    collections[f"test_collection/{collection_idx}"] = artifact
                    collection_idx += 1

    return adapter


@pytest.fixture
def client_factory(
    adapter_factory: Callable[[Optional[str], Optional[str]], InMemoryTrackingAdapter],
) -> Callable[[Optional[str], Optional[str]], InMemoryTrackingClient]:
    def _factory(
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> InMemoryTrackingClient:
        adapter = adapter_factory(experiment_id, run_id)
        return InMemoryTrackingClient(run=adapter)

    return _factory
