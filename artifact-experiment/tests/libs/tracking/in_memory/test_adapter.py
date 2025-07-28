from typing import Callable, Dict, List, Optional, Tuple
from uuid import UUID

import pytest
from artifact_experiment.libs.tracking.in_memory.adapter import (
    InMemoryRunAdapter,
)
from artifact_experiment.libs.tracking.in_memory.native_run import (
    InMemoryRun,
)
from matplotlib.figure import Figure
from numpy import ndarray


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
    adapter = InMemoryRunAdapter.build(experiment_id=experiment_id, run_id=run_id)
    assert adapter.experiment_id == experiment_id
    assert adapter.is_active is True
    if run_id is not None:
        assert adapter.run_id == run_id
    else:
        assert adapter.run_id is not None
        assert len(adapter.run_id) == standard_uuid_length
        UUID(adapter.run_id)


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
    adapter = InMemoryRunAdapter.from_native_run(native_run)
    assert adapter.experiment_id == experiment_id
    assert adapter.run_id == run_id
    assert adapter.is_active is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id",
    [("exp1", "run1")],
)
def test_stop_run(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRun, InMemoryRunAdapter]
    ],
    experiment_id: str,
    run_id: str,
):
    native_run, adapter = adapter_factory(experiment_id, run_id)
    assert native_run.is_active
    assert adapter.is_active
    adapter.stop()
    assert not native_run.is_active
    assert not adapter.is_active


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id",
    [("exp1", "run1")],
)
def test_native_context_manager(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRun, InMemoryRunAdapter]
    ],
    experiment_id: str,
    run_id: str,
):
    native_run, adapter = adapter_factory(experiment_id, run_id)
    with adapter.native() as ctx_native_run:
        assert ctx_native_run is native_run


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, score_key, score",
    [
        ("exp1", "run1", "score/1", "score_1"),
        ("exp1", "run1", "score/1", "score_2"),
        ("exp1", "run1", "score/1", "score_3"),
        ("exp1", "run1", "score/1", "score_4"),
    ],
    indirect=["score"],
)
def test_log_score(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRun, InMemoryRunAdapter]
    ],
    experiment_id: str,
    run_id: str,
    score_key: str,
    score: float,
):
    _, adapter = adapter_factory(experiment_id, run_id)
    assert adapter.n_scores == 0
    assert adapter.dict_scores == {}
    assert adapter.ls_scores == []
    adapter.log_score(path=score_key, score=score)
    assert adapter.n_scores == 1
    assert adapter.dict_scores == {score_key: score}
    assert adapter.ls_scores == [score]


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, array_key, array",
    [
        ("exp1", "run1", "array/1", "array_1"),
        ("exp1", "run1", "array/1", "array_2"),
        ("exp1", "run1", "array/1", "array_3"),
        ("exp1", "run1", "array/1", "array_4"),
    ],
    indirect=["array"],
)
def test_log_array(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRun, InMemoryRunAdapter]
    ],
    experiment_id: str,
    run_id: str,
    array_key: str,
    array: ndarray,
):
    _, adapter = adapter_factory(experiment_id, run_id)
    assert adapter.n_arrays == 0
    assert adapter.dict_arrays == {}
    assert adapter.ls_arrays == []
    adapter.log_array(path=array_key, array=array)
    assert adapter.n_arrays == 1
    assert (adapter.dict_arrays[array_key] == array).all()
    assert (adapter.ls_arrays[0] == array).all


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, plot_key, plot",
    [
        ("exp1", "run1", "plot/1", "plot_1"),
        ("exp1", "run1", "plot/1", "plot_2"),
        ("exp1", "run1", "plot/1", "plot_3"),
        ("exp1", "run1", "plot/1", "plot_4"),
    ],
    indirect=["plot"],
)
def test_log_plot(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRun, InMemoryRunAdapter]
    ],
    experiment_id: str,
    run_id: str,
    plot_key: str,
    plot: Figure,
):
    _, adapter = adapter_factory(experiment_id, run_id)
    assert adapter.n_plots == 0
    assert adapter.dict_plots == {}
    assert adapter.ls_plots == []
    adapter.log_plot(path=plot_key, plot=plot)
    assert adapter.n_plots == 1
    assert isinstance(adapter.dict_plots[plot_key], Figure)
    assert isinstance(adapter.ls_plots[0], Figure)


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, score_collection_key, score_collection",
    [
        ("exp1", "run1", "score_collection/1", "score_collection_1"),
        ("exp1", "run1", "score_collection/1", "score_collection_2"),
        ("exp1", "run1", "score_collection/1", "score_collection_3"),
        ("exp1", "run1", "score_collection/1", "score_collection_4"),
    ],
    indirect=["score_collection"],
)
def test_log_score_collection(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRun, InMemoryRunAdapter]
    ],
    experiment_id: str,
    run_id: str,
    score_collection_key: str,
    score_collection: Dict[str, float],
):
    _, adapter = adapter_factory(experiment_id, run_id)
    assert adapter.n_score_collections == 0
    assert adapter.dict_score_collections == {}
    assert adapter.ls_score_collections == []
    adapter.log_score_collection(path=score_collection_key, score_collection=score_collection)
    assert adapter.n_score_collections == 1
    assert adapter.dict_score_collections == {score_collection_key: score_collection}
    assert adapter.ls_score_collections == [score_collection]


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, array_collection_key, array_collection",
    [
        ("exp1", "run1", "array_collection/1", "array_collection_1"),
        ("exp1", "run1", "array_collection/1", "array_collection_2"),
        ("exp1", "run1", "array_collection/1", "array_collection_3"),
        ("exp1", "run1", "array_collection/1", "array_collection_4"),
    ],
    indirect=["array_collection"],
)
def test_log_array_collection(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRun, InMemoryRunAdapter]
    ],
    experiment_id: str,
    run_id: str,
    array_collection_key: str,
    array_collection: Dict[str, ndarray],
):
    _, adapter = adapter_factory(experiment_id, run_id)
    assert adapter.n_array_collections == 0
    assert adapter.dict_array_collections == {}
    assert adapter.ls_array_collections == []
    adapter.log_array_collection(path=array_collection_key, array_collection=array_collection)
    assert adapter.n_array_collections == 1
    assert (adapter.dict_array_collections[array_collection_key].keys()) == array_collection.keys()
    for key in array_collection.keys():
        assert (
            adapter.dict_array_collections[array_collection_key][key] == array_collection[key]
        ).all()


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, plot_collection_key, plot_collection",
    [
        ("exp1", "run1", "plot_collection/1", "plot_collection_1"),
        ("exp1", "run1", "plot_collection/1", "plot_collection_2"),
        ("exp1", "run1", "plot_collection/1", "plot_collection_3"),
        ("exp1", "run1", "plot_collection/1", "plot_collection_4"),
    ],
    indirect=["plot_collection"],
)
def test_log_plot_collection(
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRun, InMemoryRunAdapter]
    ],
    experiment_id: str,
    run_id: str,
    plot_collection_key: str,
    plot_collection: Dict[str, Figure],
):
    _, adapter = adapter_factory(experiment_id, run_id)
    assert adapter.n_plot_collections == 0
    assert adapter.dict_plot_collections == {}
    assert adapter.ls_plot_collections == []
    adapter.log_plot_collection(path=plot_collection_key, plot_collection=plot_collection)
    assert len(adapter.dict_plot_collections) == 1
    assert (adapter.dict_plot_collections[plot_collection_key].keys()) == plot_collection.keys()
    for key in plot_collection.keys():
        assert isinstance(adapter.dict_plot_collections[plot_collection_key][key], Figure)
        assert isinstance(plot_collection[key], Figure)


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
    adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRun, InMemoryRunAdapter]
    ],
    experiment_id: str,
    run_id: str,
    ls_file_entries: List[Dict[str, str]],
    expected_store_length: int,
):
    _, adapter = adapter_factory(None, None)
    assert len(adapter.uploaded_files) == 0
    for file_entry in ls_file_entries:
        adapter.upload(**file_entry)
    assert len(adapter.uploaded_files) == expected_store_length
    for i, file_entry in enumerate(ls_file_entries):
        assert adapter.uploaded_files[i] == file_entry


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, populated_adapter_factory, artifact_path, expected_ls_paths",
    [
        ("exp1", "run1", ["score_1"], "score_1", []),
        ("exp1", "run1", ["score_1", "score_2"], "t", ["test_score/1", "test_score/2"]),
        ("exp1", "run1", ["score_1", "score_2"], "test_score", ["test_score/1", "test_score/2"]),
        (
            "exp1",
            "run1",
            ["score_1", "score_2", "score_3"],
            "t",
            ["test_score/1", "test_score/2", "test_score/3"],
        ),
        ("exp1", "run1", ["score_1", "score_2", "score_3"], "test_score/1", ["test_score/1"]),
    ],
    indirect=["populated_adapter_factory"],
)
def test_search_score(
    populated_adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRun, InMemoryRunAdapter]
    ],
    experiment_id: str,
    run_id: str,
    artifact_path: str,
    expected_ls_paths: List[str],
):
    _, adapter = populated_adapter_factory(experiment_id, run_id)
    ls_paths = adapter.search_score_store(artifact_path=artifact_path)
    assert ls_paths == expected_ls_paths


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, populated_adapter_factory, artifact_path, expected_ls_paths",
    [
        ("exp1", "run1", ["array_1"], "array_1", []),
        ("exp1", "run1", ["array_1", "array_2"], "t", ["test_array/1", "test_array/2"]),
        ("exp1", "run1", ["array_1", "array_2"], "test_array", ["test_array/1", "test_array/2"]),
        (
            "exp1",
            "run1",
            ["array_1", "array_2", "array_3"],
            "t",
            ["test_array/1", "test_array/2", "test_array/3"],
        ),
        ("exp1", "run1", ["array_1", "array_2", "array_3"], "test_array/1", ["test_array/1"]),
    ],
    indirect=["populated_adapter_factory"],
)
def test_search_array(
    populated_adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRun, InMemoryRunAdapter]
    ],
    experiment_id: str,
    run_id: str,
    artifact_path: str,
    expected_ls_paths: List[str],
):
    _, adapter = populated_adapter_factory(experiment_id, run_id)
    ls_paths = adapter.search_array_store(artifact_path=artifact_path)
    assert ls_paths == expected_ls_paths


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, populated_adapter_factory, artifact_path, expected_ls_paths",
    [
        ("exp1", "run1", ["plot_1"], "plot_1", []),
        ("exp1", "run1", ["plot_1", "plot_2"], "t", ["test_plot/1", "test_plot/2"]),
        ("exp1", "run1", ["plot_1", "plot_2"], "test_plot", ["test_plot/1", "test_plot/2"]),
        (
            "exp1",
            "run1",
            ["plot_1", "plot_2", "plot_3"],
            "t",
            ["test_plot/1", "test_plot/2", "test_plot/3"],
        ),
        ("exp1", "run1", ["plot_1", "plot_2", "plot_3"], "test_plot/1", ["test_plot/1"]),
    ],
    indirect=["populated_adapter_factory"],
)
def test_search_plot(
    populated_adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRun, InMemoryRunAdapter]
    ],
    experiment_id: str,
    run_id: str,
    artifact_path: str,
    expected_ls_paths: List[str],
):
    _, adapter = populated_adapter_factory(experiment_id, run_id)
    ls_paths = adapter.search_plot_store(artifact_path=artifact_path)
    assert ls_paths == expected_ls_paths


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, populated_adapter_factory, artifact_path, expected_ls_paths",
    [
        ("exp1", "run1", ["score_collection_1"], "score_collection_1", []),
        (
            "exp1",
            "run1",
            ["score_collection_1", "score_collection_2"],
            "t",
            ["test_score_collection/1", "test_score_collection/2"],
        ),
        (
            "exp1",
            "run1",
            ["score_collection_1", "score_collection_2"],
            "test_score_collection",
            ["test_score_collection/1", "test_score_collection/2"],
        ),
        (
            "exp1",
            "run1",
            ["score_collection_1", "score_collection_2", "score_collection_3"],
            "t",
            ["test_score_collection/1", "test_score_collection/2", "test_score_collection/3"],
        ),
        (
            "exp1",
            "run1",
            ["score_collection_1", "score_collection_2", "score_collection_3"],
            "test_score_collection/1",
            ["test_score_collection/1"],
        ),
    ],
    indirect=["populated_adapter_factory"],
)
def test_search_score_collection(
    populated_adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRun, InMemoryRunAdapter]
    ],
    experiment_id: str,
    run_id: str,
    artifact_path: str,
    expected_ls_paths: List[str],
):
    _, adapter = populated_adapter_factory(experiment_id, run_id)
    ls_paths = adapter.search_score_collection_store(artifact_path=artifact_path)
    assert ls_paths == expected_ls_paths


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, populated_adapter_factory, artifact_path, expected_ls_paths",
    [
        ("exp1", "run1", ["array_collection_1"], "array_collection_1", []),
        (
            "exp1",
            "run1",
            ["array_collection_1", "array_collection_2"],
            "t",
            ["test_array_collection/1", "test_array_collection/2"],
        ),
        (
            "exp1",
            "run1",
            ["array_collection_1", "array_collection_2"],
            "test_array_collection",
            ["test_array_collection/1", "test_array_collection/2"],
        ),
        (
            "exp1",
            "run1",
            ["array_collection_1", "array_collection_2", "array_collection_3"],
            "t",
            ["test_array_collection/1", "test_array_collection/2", "test_array_collection/3"],
        ),
        (
            "exp1",
            "run1",
            ["array_collection_1", "array_collection_2", "array_collection_3"],
            "test_array_collection/1",
            ["test_array_collection/1"],
        ),
    ],
    indirect=["populated_adapter_factory"],
)
def test_search_array_collection(
    populated_adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRun, InMemoryRunAdapter]
    ],
    experiment_id: str,
    run_id: str,
    artifact_path: str,
    expected_ls_paths: List[str],
):
    _, adapter = populated_adapter_factory(experiment_id, run_id)
    ls_paths = adapter.search_array_collection_store(artifact_path=artifact_path)
    assert ls_paths == expected_ls_paths


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, populated_adapter_factory, artifact_path, expected_ls_paths",
    [
        ("exp1", "run1", ["plot_collection_1"], "plot_collection_1", []),
        (
            "exp1",
            "run1",
            ["plot_collection_1", "plot_collection_2"],
            "t",
            ["test_plot_collection/1", "test_plot_collection/2"],
        ),
        (
            "exp1",
            "run1",
            ["plot_collection_1", "plot_collection_2"],
            "test_plot_collection",
            ["test_plot_collection/1", "test_plot_collection/2"],
        ),
        (
            "exp1",
            "run1",
            ["plot_collection_1", "plot_collection_2", "plot_collection_3"],
            "t",
            ["test_plot_collection/1", "test_plot_collection/2", "test_plot_collection/3"],
        ),
        (
            "exp1",
            "run1",
            ["plot_collection_1", "plot_collection_2", "plot_collection_3"],
            "test_plot_collection/1",
            ["test_plot_collection/1"],
        ),
    ],
    indirect=["populated_adapter_factory"],
)
def test_search_plot_collection(
    populated_adapter_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRun, InMemoryRunAdapter]
    ],
    experiment_id: str,
    run_id: str,
    artifact_path: str,
    expected_ls_paths: List[str],
):
    _, adapter = populated_adapter_factory(experiment_id, run_id)
    ls_paths = adapter.search_plot_collection_store(artifact_path=artifact_path)
    assert ls_paths == expected_ls_paths
