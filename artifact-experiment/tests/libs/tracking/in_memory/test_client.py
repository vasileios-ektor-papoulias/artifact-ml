from typing import Callable, Dict, List, Optional

import pytest
from artifact_experiment.libs.tracking.in_memory.adapter import (
    InMemoryTrackingAdapter,
)
from artifact_experiment.libs.tracking.in_memory.client import (
    InMemoryTrackingClient,
)
from matplotlib.figure import Figure
from numpy import ndarray


@pytest.mark.parametrize(
    "experiment_id, run_id",
    [
        ("exp1", "run1"),
        ("my_experiment", "my_run"),
        ("test-experiment", "test-run"),
        ("experiment_with_underscores", "run_with_underscores"),
    ],
)
def test_init(
    client_factory: Callable[[Optional[str], Optional[str]], InMemoryTrackingClient],
    experiment_id: str,
    run_id: str,
):
    client: InMemoryTrackingClient = client_factory(experiment_id, run_id)
    assert isinstance(client._run, InMemoryTrackingAdapter)
    assert client._run.experiment_id == experiment_id
    assert client._run.run_id == run_id
    assert client._run.is_active is True


@pytest.mark.parametrize(
    "ls_scores",
    [
        ([]),
        (["score_1"]),
        (["score_1", "score_2"]),
        (["score_1", "score_3"]),
        (["score_1", "score_2", "score_3"]),
        (["score_1", "score_2", "score_3", "score_4", "score_5"]),
    ],
    indirect=True,
)
def test_log_score(
    ls_scores: List[float],
    client_factory: Callable[[Optional[str], Optional[str]], InMemoryTrackingClient],
):
    client = client_factory(None, None)
    for idx, score in enumerate(ls_scores, start=1):
        client.log_score(score=score, name=f"score_{idx}")
    assert client._run.n_scores == len(ls_scores)
    assert client._run.ls_scores == ls_scores


@pytest.mark.parametrize(
    "ls_arrays",
    [
        ([]),
        (["array_1"]),
        (["array_1", "array_2"]),
        (["array_1", "array_3"]),
        (["array_1", "array_2", "array_3"]),
        (["array_1", "array_2", "array_3", "array_4", "array_5"]),
    ],
    indirect=True,
)
def test_log_array(
    ls_arrays: List[ndarray],
    client_factory: Callable[[Optional[str], Optional[str]], InMemoryTrackingClient],
):
    client = client_factory(None, None)
    for idx, array in enumerate(ls_arrays, start=1):
        client.log_array(array=array, name=f"array_{idx}")
    assert client._run.n_arrays == len(ls_arrays)
    assert client._run.ls_arrays == ls_arrays


@pytest.mark.parametrize(
    "ls_plots",
    [
        ([]),
        (["plot_1"]),
        (["plot_1", "plot_2"]),
        (["plot_1", "plot_3"]),
        (["plot_1", "plot_2", "plot_3"]),
        (["plot_1", "plot_2", "plot_3", "plot_4", "plot_5"]),
    ],
    indirect=True,
)
def test_log_plot(
    ls_plots: List[Figure],
    client_factory: Callable[[Optional[str], Optional[str]], InMemoryTrackingClient],
):
    client = client_factory(None, None)
    for idx, plot in enumerate(ls_plots, start=1):
        client.log_plot(plot=plot, name=f"plot_{idx}")
    assert client._run.n_plots == len(ls_plots)
    assert client._run.ls_plots == ls_plots


@pytest.mark.parametrize(
    "ls_score_collections",
    [
        ([]),
        (["score_collection_1"]),
        (["score_collection_1", "score_collection_2"]),
        (["score_collection_1", "score_collection_3"]),
        (["score_collection_1", "score_collection_2", "score_collection_3"]),
        (
            [
                "score_collection_1",
                "score_collection_2",
                "score_collection_3",
                "score_collection_4",
                "score_collection_5",
            ]
        ),
    ],
    indirect=True,
)
def test_log_score_collection(
    ls_score_collections: List[Dict[str, float]],
    client_factory: Callable[[Optional[str], Optional[str]], InMemoryTrackingClient],
):
    client = client_factory(None, None)
    for idx, score_collection in enumerate(ls_score_collections, start=1):
        client.log_score_collection(
            score_collection=score_collection, name=f"score_collection_{idx}"
        )
    assert client._run.n_score_collections == len(ls_score_collections)
    assert client._run.ls_score_collections == ls_score_collections


@pytest.mark.parametrize(
    "ls_array_collections",
    [
        ([]),
        (["array_collection_1"]),
        (["array_collection_1", "array_collection_2"]),
        (["array_collection_1", "array_collection_3"]),
        (["array_collection_1", "array_collection_2", "array_collection_3"]),
        (
            [
                "array_collection_1",
                "array_collection_2",
                "array_collection_3",
                "array_collection_4",
                "array_collection_5",
            ]
        ),
    ],
    indirect=True,
)
def test_log_array_collection(
    ls_array_collections: List[Dict[str, ndarray]],
    client_factory: Callable[[Optional[str], Optional[str]], InMemoryTrackingClient],
):
    client = client_factory(None, None)
    for idx, array_collection in enumerate(ls_array_collections, start=1):
        client.log_array_collection(
            array_collection=array_collection, name=f"array_collection_{idx}"
        )
    assert client._run.n_array_collections == len(ls_array_collections)
    assert client._run.ls_array_collections == ls_array_collections


@pytest.mark.parametrize(
    "ls_plot_collections",
    [
        ([]),
        (["plot_collection_1"]),
        (["plot_collection_1", "plot_collection_2"]),
        (["plot_collection_1", "plot_collection_3"]),
        (["plot_collection_1", "plot_collection_2", "plot_collection_3"]),
        (
            [
                "plot_collection_1",
                "plot_collection_2",
                "plot_collection_3",
                "plot_collection_4",
                "plot_collection_5",
            ]
        ),
    ],
    indirect=True,
)
def test_log_plot_collection(
    ls_plot_collections: List[Dict[str, Figure]],
    client_factory: Callable[[Optional[str], Optional[str]], InMemoryTrackingClient],
):
    client = client_factory(None, None)
    for idx, plot_collection in enumerate(ls_plot_collections, start=1):
        client.log_plot_collection(plot_collection=plot_collection, name=f"plot_collection_{idx}")
    assert client._run.n_plot_collections == len(ls_plot_collections)
    assert client._run.ls_plot_collections == ls_plot_collections


@pytest.mark.parametrize(
    "path_source,dir_target",
    [
        ("/test/path", "uploads"),
        ("/data/file.pkl", "models"),
        ("/logs/run.log", "logs"),
    ],
)
def test_upload_delegation(
    client_factory: Callable[[Optional[str], Optional[str]], InMemoryTrackingClient],
    path_source: str,
    dir_target: str,
):
    client: InMemoryTrackingClient = client_factory(None, None)
    assert len(client.uploaded_files) == 0
    client.upload(path_source=path_source, dir_target=dir_target)
    assert len(client.uploaded_files) == 1
    expected_entry = {"path_source": path_source, "dir_target": dir_target}
    assert client.uploaded_files[0] == expected_entry
