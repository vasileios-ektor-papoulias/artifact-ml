from typing import Callable, Dict, List, Optional, Tuple
from uuid import UUID

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
from matplotlib.figure import Figure
from numpy import ndarray


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id",
    [("exp1", "run1")],
)
def test_init(
    client_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRunAdapter, InMemoryTrackingClient]
    ],
    experiment_id: str,
    run_id: str,
):
    adapter, client = client_factory(experiment_id, run_id)
    assert isinstance(client, InMemoryTrackingClient)
    assert client.run is adapter
    assert client.run.experiment_id == experiment_id
    assert client.run.run_id == run_id
    assert client.run.is_active is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id",
    [("exp1", "run1")],
)
def test_from_run(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRun, InMemoryRunAdapter]
    ],
    experiment_id: str,
    run_id: str,
):
    _, adapter = adapter_factory(experiment_id, run_id)
    client = InMemoryTrackingClient.from_run(run=adapter)
    assert isinstance(client, InMemoryTrackingClient)
    assert client.run is adapter
    assert client.run.experiment_id == experiment_id
    assert client.run.run_id == run_id
    assert client.run.is_active is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id",
    [("exp1", "run1")],
)
def test_from_native_run(
    native_run_factory: Callable[[Optional[str], Optional[str]], InMemoryRun],
    experiment_id: str,
    run_id: str,
):
    native_run = native_run_factory(experiment_id, run_id)
    client = InMemoryTrackingClient.from_native_run(native_run=native_run)
    assert isinstance(client, InMemoryTrackingClient)
    assert client.run.experiment_id == experiment_id
    assert client.run.run_id == run_id
    assert client.run.is_active is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id",
    [
        ("exp1", "run1"),
        ("exp1", None),
    ],
)
def test_build(
    standard_uuid_length: int,
    experiment_id: str,
    run_id: Optional[str],
):
    client = InMemoryTrackingClient.build(experiment_id=experiment_id, run_id=run_id)
    assert isinstance(client, InMemoryTrackingClient)
    assert client.run.is_active is True
    assert client.run.experiment_id == experiment_id
    if run_id is not None:
        assert client.run.run_id == run_id
    else:
        assert client.run.run_id is not None
        assert len(client.run.run_id) == standard_uuid_length
        UUID(client.run.run_id)


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_scores",
    [
        ("exp1", "run1", []),
        ("exp1", "run1", ["score_1"]),
        ("exp1", "run1", ["score_1", "score_2"]),
        ("exp1", "run1", ["score_1", "score_3"]),
        ("exp1", "run1", ["score_1", "score_2", "score_3"]),
        ("exp1", "run1", ["score_1", "score_2", "score_3", "score_4", "score_5"]),
    ],
    indirect=["ls_scores"],
)
def test_log_score(
    client_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRunAdapter, InMemoryTrackingClient]
    ],
    experiment_id: str,
    run_id: str,
    ls_scores: List[float],
):
    adapter, client = client_factory(experiment_id, run_id)
    for idx, score in enumerate(ls_scores, start=1):
        client.log_score(score=score, name=f"score_{idx}")
    assert adapter.n_scores == len(ls_scores)
    assert adapter.ls_scores == ls_scores


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_arrays",
    [
        ("exp1", "run1", []),
        ("exp1", "run1", ["array_1"]),
        ("exp1", "run1", ["array_1", "array_2"]),
        ("exp1", "run1", ["array_1", "array_3"]),
        ("exp1", "run1", ["array_1", "array_2", "array_3"]),
        ("exp1", "run1", ["array_1", "array_2", "array_3", "array_4", "array_5"]),
    ],
    indirect=["ls_arrays"],
)
def test_log_array(
    client_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRunAdapter, InMemoryTrackingClient]
    ],
    experiment_id: str,
    run_id: str,
    ls_arrays: List[ndarray],
):
    adapter, client = client_factory(experiment_id, run_id)
    for idx, array in enumerate(ls_arrays):
        client.log_array(array=array, name=f"array_{idx + 1}")
    assert adapter.n_arrays == len(ls_arrays)
    for idx, (array_key, array_value) in enumerate(adapter.dict_arrays.items()):
        assert array_key == f"{experiment_id}/{run_id}/arrays/array_{idx + 1}/1"
        assert (array_value == ls_arrays[idx]).all


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_plots",
    [
        ("exp1", "run1", []),
        ("exp1", "run1", ["plot_1"]),
        ("exp1", "run1", ["plot_1", "plot_2"]),
        ("exp1", "run1", ["plot_1", "plot_3"]),
        ("exp1", "run1", ["plot_1", "plot_2", "plot_3"]),
        ("exp1", "run1", ["plot_1", "plot_2", "plot_3", "plot_4", "plot_5"]),
    ],
    indirect=["ls_plots"],
)
def test_log_plot(
    client_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRunAdapter, InMemoryTrackingClient]
    ],
    experiment_id: str,
    run_id: str,
    ls_plots: List[Figure],
):
    adapter, client = client_factory(experiment_id, run_id)
    for idx, plot in enumerate(ls_plots, start=1):
        client.log_plot(plot=plot, name=f"plot_{idx}")
    assert adapter.n_plots == len(ls_plots)
    for idx, (plot_key, plot) in enumerate(adapter.dict_plots.items()):
        assert plot_key == f"{experiment_id}/{run_id}/plots/plot_{idx + 1}/1"
        assert isinstance(plot, Figure)


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_score_collections",
    [
        ("exp1", "run1", []),
        ("exp1", "run1", ["score_collection_1"]),
        ("exp1", "run1", ["score_collection_1", "score_collection_2"]),
        ("exp1", "run1", ["score_collection_1", "score_collection_3"]),
        ("exp1", "run1", ["score_collection_1", "score_collection_2", "score_collection_3"]),
        (
            "exp1",
            "run1",
            [
                "score_collection_1",
                "score_collection_2",
                "score_collection_3",
                "score_collection_4",
                "score_collection_5",
            ],
        ),
    ],
    indirect=["ls_score_collections"],
)
def test_log_score_collection(
    experiment_id: str,
    run_id: str,
    ls_score_collections: List[Dict[str, float]],
    client_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRunAdapter, InMemoryTrackingClient]
    ],
):
    adapter, client = client_factory(experiment_id, run_id)
    for idx, score_collection in enumerate(ls_score_collections, start=1):
        client.log_score_collection(
            score_collection=score_collection, name=f"score_collection_{idx}"
        )
    assert adapter.n_score_collections == len(ls_score_collections)
    assert adapter.ls_score_collections == ls_score_collections


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_array_collections",
    [
        ("exp1", "run1", []),
        ("exp1", "run1", ["array_collection_1"]),
        ("exp1", "run1", ["array_collection_1", "array_collection_2"]),
        ("exp1", "run1", ["array_collection_1", "array_collection_3"]),
        ("exp1", "run1", ["array_collection_1", "array_collection_2", "array_collection_3"]),
        (
            "exp1",
            "run1",
            [
                "array_collection_1",
                "array_collection_2",
                "array_collection_3",
                "array_collection_4",
                "array_collection_5",
            ],
        ),
    ],
    indirect=["ls_array_collections"],
)
def test_log_array_collection(
    client_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRunAdapter, InMemoryTrackingClient]
    ],
    experiment_id: str,
    run_id: str,
    ls_array_collections: List[Dict[str, ndarray]],
):
    adapter, client = client_factory(experiment_id, run_id)
    for idx, array_collection in enumerate(ls_array_collections):
        client.log_array_collection(array_collection=array_collection, name=f"array_{idx + 1}")
    assert adapter.n_array_collections == len(ls_array_collections)
    for idx, (array_collection_key, array_collection_value) in enumerate(
        adapter.dict_array_collections.items()
    ):
        assert (
            array_collection_key == f"{experiment_id}/{run_id}/array_collections/array_{idx + 1}/1"
        )
        for actual_array, expected_array in zip(
            array_collection_value.values(), ls_array_collections[idx]
        ):
            assert (actual_array == expected_array).all


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_plot_collections",
    [
        ("exp1", "run1", []),
        ("exp1", "run1", ["plot_collection_1"]),
        ("exp1", "run1", ["plot_collection_1", "plot_collection_2"]),
        ("exp1", "run1", ["plot_collection_1", "plot_collection_3"]),
        ("exp1", "run1", ["plot_collection_1", "plot_collection_2", "plot_collection_3"]),
        (
            "exp1",
            "run1",
            [
                "plot_collection_1",
                "plot_collection_2",
                "plot_collection_3",
                "plot_collection_4",
                "plot_collection_5",
            ],
        ),
    ],
    indirect=["ls_plot_collections"],
)
def test_log_plot_collection(
    client_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRunAdapter, InMemoryTrackingClient]
    ],
    experiment_id: str,
    run_id: str,
    ls_plot_collections: List[Dict[str, Figure]],
):
    adapter, client = client_factory(experiment_id, run_id)
    for idx, plot_collection in enumerate(ls_plot_collections):
        client.log_plot_collection(plot_collection=plot_collection, name=f"plot_{idx + 1}")
    assert adapter.n_plot_collections == len(ls_plot_collections)
    for idx, (plot_collection_key, plot_collection_value) in enumerate(
        adapter.dict_plot_collections.items()
    ):
        assert plot_collection_key == f"{experiment_id}/{run_id}/plot_collections/plot_{idx + 1}/1"
        for plot in plot_collection_value.values():
            isinstance(plot, Figure)


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_file_entries, expected_store_length",
    [
        ("exp1", "run1", [], 0),
        ("exp1", "run1", [{"path_source": "/test/path1", "dir_target": "uploads"}], 1),
        (
            "exp1",
            "run1",
            [
                {"path_source": "/test/path1", "dir_target": "uploads"},
                {"path_source": "/another/path", "dir_target": "data"},
            ],
            2,
        ),
    ],
)
def test_upload(
    client_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRunAdapter, InMemoryTrackingClient]
    ],
    experiment_id: str,
    run_id: str,
    ls_file_entries: List[Dict[str, str]],
    expected_store_length: int,
):
    adapter, client = client_factory(experiment_id, run_id)
    assert len(adapter.uploaded_files) == 0
    for file_entry in ls_file_entries:
        client.upload(**file_entry)
    assert len(adapter.uploaded_files) == expected_store_length
    for i, file_entry in enumerate(ls_file_entries):
        assert adapter.uploaded_files[i] == file_entry
