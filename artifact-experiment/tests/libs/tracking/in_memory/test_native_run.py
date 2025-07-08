from typing import Callable, Dict, List, Optional

import numpy as np
import pytest
from artifact_experiment.libs.tracking.in_memory.native_run import (
    InMemoryRun,
)
from matplotlib.figure import Figure


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
    "score_key,score_value",
    [
        ("accuracy/1", 0.95),
        ("loss/1", 0.05),
        ("f1_score/2", 0.87),
        ("precision/10", 0.92),
    ],
)
def test_dict_scores(
    native_run_factory: Callable[[Optional[str], Optional[str]], InMemoryRun],
    score_key: str,
    score_value: float,
):
    native_run: InMemoryRun = native_run_factory(None, None)
    scores_dict: Dict[str, float] = native_run.dict_scores
    assert len(scores_dict) == 0
    scores_dict[score_key] = score_value
    assert len(native_run.dict_scores) == 1
    assert native_run.dict_scores[score_key] == score_value


def test_dict_arrays(
    native_run_factory: Callable[[Optional[str], Optional[str]], InMemoryRun],
    array_1: np.ndarray,
):
    native_run: InMemoryRun = native_run_factory(None, None)
    arrays_dict: Dict[str, np.ndarray] = native_run.dict_arrays
    assert len(arrays_dict) == 0
    arrays_dict["test_array/1"] = array_1
    assert len(native_run.dict_arrays) == 1
    np.testing.assert_array_equal(native_run.dict_arrays["test_array/1"], array_1)


def test_dict_plots(
    native_run_factory: Callable[[Optional[str], Optional[str]], InMemoryRun],
    plot_1: Figure,
):
    native_run: InMemoryRun = native_run_factory(None, None)
    plots_dict: Dict[str, Figure] = native_run.dict_plots
    assert len(plots_dict) == 0
    plots_dict["test_plot/1"] = plot_1
    assert len(native_run.dict_plots) == 1
    assert native_run.dict_plots["test_plot/1"] is plot_1


def test_dict_score_collections(
    native_run_factory: Callable[[Optional[str], Optional[str]], InMemoryRun],
    score_collection_1: Dict[str, float],
):
    native_run: InMemoryRun = native_run_factory(None, None)
    collections_dict: Dict[str, Dict[str, float]] = native_run.dict_score_collections
    assert len(collections_dict) == 0
    collections_dict["test_collection/1"] = score_collection_1
    assert len(native_run.dict_score_collections) == 1
    assert native_run.dict_score_collections["test_collection/1"] == score_collection_1


def test_dict_array_collections(
    native_run_factory: Callable[[Optional[str], Optional[str]], InMemoryRun],
    array_collection_1: Dict[str, np.ndarray],
):
    native_run: InMemoryRun = native_run_factory(None, None)
    collections_dict: Dict[str, Dict[str, np.ndarray]] = native_run.dict_array_collections
    assert len(collections_dict) == 0
    collections_dict["test_collection/1"] = array_collection_1
    assert len(native_run.dict_array_collections) == 1
    stored_collection = native_run.dict_array_collections["test_collection/1"]
    for key, array in array_collection_1.items():
        np.testing.assert_array_equal(stored_collection[key], array)


def test_dict_plot_collections(
    native_run_factory: Callable[[Optional[str], Optional[str]], InMemoryRun],
    plot_collection_1: Dict[str, Figure],
):
    native_run: InMemoryRun = native_run_factory(None, None)
    collections_dict: Dict[str, Dict[str, Figure]] = native_run.dict_plot_collections
    assert len(collections_dict) == 0
    collections_dict["test_collection/1"] = plot_collection_1
    assert len(native_run.dict_plot_collections) == 1
    stored_collection = native_run.dict_plot_collections["test_collection/1"]
    for key, plot in plot_collection_1.items():
        assert stored_collection[key] is plot


def test_uploaded_files(
    native_run_factory: Callable[[Optional[str], Optional[str]], InMemoryRun],
):
    native_run: InMemoryRun = native_run_factory(None, None)
    files_list: List[Dict[str, str]] = native_run.uploaded_files
    assert len(files_list) == 0
    file_entry = {"path_source": "/test/path", "dir_target": "uploads"}
    files_list.append(file_entry)
    assert len(native_run.uploaded_files) == 1
    assert native_run.uploaded_files[0] == file_entry


@pytest.mark.parametrize(
    "property_name,expected_type",
    [
        ("experiment_id", str),
        ("run_id", str),
        ("is_active", bool),
        ("dict_scores", dict),
        ("dict_arrays", dict),
        ("dict_plots", dict),
        ("dict_score_collections", dict),
        ("dict_array_collections", dict),
        ("dict_plot_collections", dict),
        ("uploaded_files", list),
    ],
)
def test_property_types(
    native_run_factory,
    property_name: str,
    expected_type: type,
):
    native_run: InMemoryRun = native_run_factory(None, None)
    property_value = getattr(native_run, property_name)
    assert isinstance(property_value, expected_type)
