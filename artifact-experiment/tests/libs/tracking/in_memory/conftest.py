from typing import Callable, Dict, Optional

import matplotlib
import numpy as np
import pytest
from artifact_experiment.libs.tracking.in_memory.adapter import (
    InMemoryTrackingAdapter,
)
from artifact_experiment.libs.tracking.in_memory.native_run import InMemoryNativeRun
from matplotlib.figure import Figure

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
    request, adapter_factory: Callable[[Optional[str], Optional[str]], InMemoryTrackingAdapter]
) -> InMemoryTrackingAdapter:
    adapter = adapter_factory(None, None)
    fixture_names = request.getfixturevalue(request.param)

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
def sample_score() -> float:
    return 0.85


@pytest.fixture
def sample_array() -> np.ndarray:
    return np.array([1, 2, 3, 4, 5])


@pytest.fixture
def sample_plot() -> Figure:
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.plot([1, 2, 3], [4, 5, 6])
    return fig


@pytest.fixture
def sample_score_collection() -> Dict[str, float]:
    return {"accuracy": 0.95, "precision": 0.87, "recall": 0.92}


@pytest.fixture
def sample_array_collection() -> Dict[str, np.ndarray]:
    return {"predictions": np.array([1, 0, 1, 1, 0]), "targets": np.array([1, 0, 1, 0, 1])}


@pytest.fixture
def sample_plot_collection() -> Dict[str, Figure]:
    fig1 = Figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot([1, 2], [1, 2])

    fig2 = Figure()
    ax2 = fig2.add_subplot(111)
    ax2.scatter([1, 2], [1, 2])

    return {"line_plot": fig1, "scatter_plot": fig2}


@pytest.fixture
def scores_only():
    return ["sample_score"]


@pytest.fixture
def arrays_only():
    return ["sample_array"]


@pytest.fixture
def plots_only():
    return ["sample_plot"]


@pytest.fixture
def score_collections_only():
    return ["sample_score_collection"]


@pytest.fixture
def array_collections_only():
    return ["sample_array_collection"]


@pytest.fixture
def plot_collections_only():
    return ["sample_plot_collection"]


@pytest.fixture
def scores_and_arrays():
    return ["sample_score", "sample_array"]


@pytest.fixture
def scores_and_plots():
    return ["sample_score", "sample_plot"]


@pytest.fixture
def arrays_and_plots():
    return ["sample_array", "sample_plot"]


@pytest.fixture
def all_primitives():
    return ["sample_score", "sample_array", "sample_plot"]


@pytest.fixture
def all_collections():
    return ["sample_score_collection", "sample_array_collection", "sample_plot_collection"]


@pytest.fixture
def mixed_artifacts():
    return ["sample_score", "sample_array", "sample_plot", "sample_score_collection"]


@pytest.fixture
def all_artifacts():
    return [
        "sample_score",
        "sample_array",
        "sample_plot",
        "sample_score_collection",
        "sample_array_collection",
        "sample_plot_collection",
    ]
