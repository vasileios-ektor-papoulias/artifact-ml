from typing import Callable, Dict, List, Optional

import pytest
from artifact_experiment.libs.tracking.in_memory.native_run import (
    InMemoryRun,
)
from matplotlib.figure import Figure
from numpy import ndarray


@pytest.mark.parametrize(
    "experiment_id,run_id",
    [
        ("exp1", "run1"),
        ("my_experiment", "my_run"),
        ("test-experiment", "test-run"),
        ("experiment_with_underscores", "run_with_underscores"),
    ],
)
def test_init(
    experiment_id: str,
    run_id: str,
    native_run_factory: Callable[[Optional[str], Optional[str]], InMemoryRun],
):
    native_run: InMemoryRun = native_run_factory(experiment_id, run_id)
    assert native_run.experiment_id == experiment_id
    assert native_run.run_id == run_id
    assert native_run.is_active is True
    assert len(native_run.dict_scores) == 0
    assert len(native_run.dict_arrays) == 0
    assert len(native_run.dict_plots) == 0
    assert len(native_run.dict_score_collections) == 0
    assert len(native_run.dict_array_collections) == 0
    assert len(native_run.dict_plot_collections) == 0
    assert len(native_run.uploaded_files) == 0


@pytest.mark.parametrize(
    "initial_state,new_state",
    [
        (True, False),
        (False, True),
        (True, True),
        (False, False),
    ],
)
def test_is_active_property(
    native_run_factory: Callable[[Optional[str], Optional[str]], InMemoryRun],
    initial_state: bool,
    new_state: bool,
):
    native_run: InMemoryRun = native_run_factory(None, None)
    assert native_run.is_active is True
    native_run.is_active = initial_state
    assert native_run.is_active == initial_state
    native_run.is_active = new_state
    assert native_run.is_active == new_state


@pytest.mark.parametrize(
    "score_key, score",
    [
        ("accuracy/1", "score_1"),
        ("loss/1", "score_2"),
        ("f1_score/2", "score_3"),
        ("precision/10", "score_4"),
    ],
    indirect=["score"],
)
def test_log_score(
    native_run_factory: Callable[[Optional[str], Optional[str]], InMemoryRun],
    score_key: str,
    score: float,
):
    native_run: InMemoryRun = native_run_factory(None, None)
    scores_dict = native_run.dict_scores
    assert len(scores_dict) == 0
    native_run.log_score(path=score_key, score=score)
    assert len(native_run.dict_scores) == 1
    assert native_run.dict_scores[score_key] == score


@pytest.mark.parametrize(
    "array_key, array",
    [
        ("array/1", "array_1"),
        ("array/1", "array_2"),
        ("array/2", "array_3"),
        ("array/10", "array_4"),
    ],
    indirect=["array"],
)
def test_log_array(
    native_run_factory: Callable[[Optional[str], Optional[str]], InMemoryRun],
    array_key: str,
    array: ndarray,
):
    native_run: InMemoryRun = native_run_factory(None, None)
    arrays_dict = native_run.dict_arrays
    assert len(arrays_dict) == 0
    native_run.log_array(path=array_key, array=array)
    assert len(native_run.dict_arrays) == 1
    assert (native_run.dict_arrays[array_key] == array).all()


@pytest.mark.parametrize(
    "plot_key, plot",
    [
        ("plot/1", "plot_1"),
        ("plot/1", "plot_2"),
        ("plot/2", "plot_3"),
        ("plot/10", "plot_4"),
    ],
    indirect=["plot"],
)
def test_log_plot(
    native_run_factory: Callable[[Optional[str], Optional[str]], InMemoryRun],
    plot_key: str,
    plot: Figure,
):
    native_run: InMemoryRun = native_run_factory(None, None)
    plots_dict = native_run.dict_plots
    assert len(plots_dict) == 0
    native_run.log_plot(path=plot_key, plot=plot)
    assert len(native_run.dict_plots) == 1


@pytest.mark.parametrize(
    "score_collection_key, score_collection",
    [
        ("score_collection/1", "score_collection_1"),
        ("score_collection/1", "score_collection_2"),
        ("score_collection/2", "score_collection_3"),
        ("score_collection/10", "score_collection_4"),
    ],
    indirect=["score_collection"],
)
def test_log_score_collection(
    native_run_factory: Callable[[Optional[str], Optional[str]], InMemoryRun],
    score_collection_key: str,
    score_collection: Dict[str, float],
):
    native_run: InMemoryRun = native_run_factory(None, None)
    scores_dict = native_run.dict_scores
    assert len(scores_dict) == 0
    native_run.log_score_collection(path=score_collection_key, score_collection=score_collection)
    assert len(native_run.dict_score_collections) == 1
    assert (
        native_run.dict_score_collections[score_collection_key].keys()
    ) == score_collection.keys()
    for key in score_collection.keys():
        assert native_run.dict_score_collections[score_collection_key][key] == score_collection[key]


@pytest.mark.parametrize(
    "array_collection_key, array_collection",
    [
        ("array_collection/1", "array_collection_1"),
        ("array_collection/1", "array_collection_2"),
        ("array_collection/2", "array_collection_3"),
        ("array_collection/10", "array_collection_4"),
    ],
    indirect=["array_collection"],
)
def test_log_array_collection(
    native_run_factory: Callable[[Optional[str], Optional[str]], InMemoryRun],
    array_collection_key: str,
    array_collection: Dict[str, ndarray],
):
    native_run: InMemoryRun = native_run_factory(None, None)
    arrays_dict = native_run.dict_arrays
    assert len(arrays_dict) == 0
    native_run.log_array_collection(path=array_collection_key, array_collection=array_collection)
    assert len(native_run.dict_array_collections) == 1
    assert (
        native_run.dict_array_collections[array_collection_key].keys()
    ) == array_collection.keys()
    for key in array_collection.keys():
        assert (
            native_run.dict_array_collections[array_collection_key][key] == array_collection[key]
        ).all()


@pytest.mark.parametrize(
    "plot_collection_key, plot_collection",
    [
        ("plot_collection/1", "plot_collection_1"),
        ("plot_collection/1", "plot_collection_2"),
        ("plot_collection/2", "plot_collection_3"),
        ("plot_collection/10", "plot_collection_4"),
    ],
    indirect=["plot_collection"],
)
def test_log_plot_collection(
    native_run_factory: Callable[[Optional[str], Optional[str]], InMemoryRun],
    plot_collection_key: str,
    plot_collection: Dict[str, Figure],
):
    native_run: InMemoryRun = native_run_factory(None, None)
    plots_dict = native_run.dict_plots
    assert len(plots_dict) == 0
    native_run.log_plot_collection(path=plot_collection_key, plot_collection=plot_collection)
    assert len(native_run.dict_plot_collections) == 1
    assert (native_run.dict_plot_collections[plot_collection_key].keys()) == plot_collection.keys()
    for key in plot_collection.keys():
        assert isinstance(native_run.dict_plot_collections[plot_collection_key][key], Figure)
        assert isinstance(plot_collection[key], Figure)


@pytest.mark.parametrize(
    "ls_file_entries, expected_store_length",
    [
        ([], 0),
        ([{"path_source": "/test/path1", "dir_target": "uploads"}], 1),
        (
            [
                {"path_source": "/test/path1", "dir_target": "uploads"},
                {"path_source": "/another/path", "dir_target": "data"},
            ],
            2,
        ),
    ],
)
def test_upload(
    native_run_factory: Callable[[Optional[str], Optional[str]], InMemoryRun],
    ls_file_entries: List[Dict[str, str]],
    expected_store_length: int,
):
    native_run = native_run_factory(None, None)
    assert len(native_run.uploaded_files) == 0
    for file_entry in ls_file_entries:
        native_run.upload(**file_entry)
    assert len(native_run.uploaded_files) == expected_store_length
    for i, file_entry in enumerate(ls_file_entries):
        assert native_run.uploaded_files[i] == file_entry
