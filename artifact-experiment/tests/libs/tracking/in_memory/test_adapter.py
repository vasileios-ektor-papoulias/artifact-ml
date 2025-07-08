from typing import Callable, Optional
from uuid import UUID

import pytest
from artifact_experiment.libs.tracking.in_memory.adapter import (
    InMemoryRunAdapter,
)
from artifact_experiment.libs.tracking.in_memory.native_run import (
    InMemoryRun,
)

STANDARD_UUID_LENGTH = 36


@pytest.mark.parametrize(
    "experiment_id, run_id",
    [
        ("exp1", "run1"),
        ("my_experiment", "my_run"),
        ("test-experiment", "test-run"),
        ("experiment_with_underscores", "run_with_underscores"),
        ("exp1", None),
        ("my_experiment", None),
        ("test-experiment", None),
        ("experiment_with_underscores", None),
    ],
)
def test_build(
    experiment_id: str,
    run_id: Optional[str],
):
    adapter: InMemoryRunAdapter = InMemoryRunAdapter.build(
        experiment_id=experiment_id, run_id=run_id
    )
    assert adapter.experiment_id == experiment_id
    assert adapter.is_active is True

    if run_id is not None:
        assert adapter.run_id == run_id
    else:
        assert adapter.run_id is not None
        assert len(adapter.run_id) == STANDARD_UUID_LENGTH
        UUID(adapter.run_id)


def test_from_native_run(
    native_run_factory: Callable[[Optional[str], Optional[str]], InMemoryRun],
):
    native_run: InMemoryRun = native_run_factory("test_exp", "test_run")
    adapter: InMemoryRunAdapter = InMemoryRunAdapter.from_native_run(native_run)
    assert adapter.experiment_id == "test_exp"
    assert adapter.run_id == "test_run"
    assert adapter.is_active is True


def test_stop_run(
    adapter_factory: Callable[[Optional[str], Optional[str]], InMemoryRunAdapter],
):
    adapter: InMemoryRunAdapter = adapter_factory(None, None)
    assert adapter.is_active is True
    adapter.stop()
    assert adapter.is_active is False


def test_native_context_manager(
    adapter_factory: Callable[[Optional[str], Optional[str]], InMemoryRunAdapter],
):
    adapter: InMemoryRunAdapter = adapter_factory(None, None)
    with adapter.native() as native_run:
        assert isinstance(native_run, InMemoryRun)
        assert native_run.experiment_id == adapter.experiment_id
        assert native_run.run_id == adapter.run_id


def test_property_delegation(
    adapter_factory: Callable[[Optional[str], Optional[str]], InMemoryRunAdapter],
):
    adapter: InMemoryRunAdapter = adapter_factory(None, None)
    native_run: InMemoryRun = adapter._native_run
    assert adapter.experiment_id == native_run.experiment_id
    assert adapter.run_id == native_run.run_id
    assert adapter.is_active == native_run.is_active
    native_run.is_active = False
    assert adapter.is_active is False


@pytest.mark.parametrize(
    "populated_adapter, expected_counts",
    [
        (["score_1"], (1, 0, 0, 0, 0, 0)),
        (["array_1"], (0, 1, 0, 0, 0, 0)),
        (["plot_1"], (0, 0, 1, 0, 0, 0)),
        (["score_collection_1"], (0, 0, 0, 1, 0, 0)),
        (["array_collection_1"], (0, 0, 0, 0, 1, 0)),
        (["plot_collection_1"], (0, 0, 0, 0, 0, 1)),
        (["score_1", "array_1"], (1, 1, 0, 0, 0, 0)),
        (["score_1", "array_1", "plot_1"], (1, 1, 1, 0, 0, 0)),
        (["score_collection_1", "array_collection_1", "plot_collection_1"], (0, 0, 0, 1, 1, 1)),
        (["score_1", "array_1", "plot_1", "score_collection_1"], (1, 1, 1, 1, 0, 0)),
        (
            [
                "score_1",
                "array_1",
                "plot_1",
                "score_collection_1",
                "array_collection_1",
                "plot_collection_1",
            ],
            (1, 1, 1, 1, 1, 1),
        ),
    ],
    indirect=["populated_adapter"],
)
def test_cache(
    populated_adapter: InMemoryRunAdapter,
    expected_counts: tuple,
):
    exp_scores, exp_arrays, exp_plots, exp_score_coll, exp_array_coll, exp_plot_coll = (
        expected_counts
    )

    assert isinstance(populated_adapter.dict_scores, dict)
    assert len(populated_adapter.dict_scores) == exp_scores
    assert isinstance(populated_adapter.ls_scores, list)
    assert len(populated_adapter.ls_scores) == exp_scores
    assert isinstance(populated_adapter.n_scores, int)
    assert populated_adapter.n_scores == exp_scores

    assert isinstance(populated_adapter.dict_arrays, dict)
    assert len(populated_adapter.dict_arrays) == exp_arrays
    assert isinstance(populated_adapter.ls_arrays, list)
    assert len(populated_adapter.ls_arrays) == exp_arrays
    assert isinstance(populated_adapter.n_arrays, int)
    assert populated_adapter.n_arrays == exp_arrays

    assert isinstance(populated_adapter.dict_plots, dict)
    assert len(populated_adapter.dict_plots) == exp_plots
    assert isinstance(populated_adapter.ls_plots, list)
    assert len(populated_adapter.ls_plots) == exp_plots
    assert isinstance(populated_adapter.n_plots, int)
    assert populated_adapter.n_plots == exp_plots

    score_collections = populated_adapter.dict_score_collections
    assert isinstance(score_collections, dict)
    assert len(score_collections) == exp_score_coll
    assert isinstance(populated_adapter.ls_score_collections, list)
    assert len(populated_adapter.ls_score_collections) == exp_score_coll
    assert isinstance(populated_adapter.n_score_collections, int)
    assert populated_adapter.n_score_collections == exp_score_coll

    array_collections = populated_adapter.dict_array_collections
    assert isinstance(array_collections, dict)
    assert len(array_collections) == exp_array_coll
    assert isinstance(populated_adapter.ls_array_collections, list)
    assert len(populated_adapter.ls_array_collections) == exp_array_coll
    assert isinstance(populated_adapter.n_array_collections, int)
    assert populated_adapter.n_array_collections == exp_array_coll

    plot_collections = populated_adapter.dict_plot_collections
    assert isinstance(plot_collections, dict)
    assert len(plot_collections) == exp_plot_coll
    assert isinstance(populated_adapter.ls_plot_collections, list)
    assert len(populated_adapter.ls_plot_collections) == exp_plot_coll
    assert isinstance(populated_adapter.n_plot_collections, int)
    assert populated_adapter.n_plot_collections == exp_plot_coll

    assert isinstance(populated_adapter.uploaded_files, list)


@pytest.mark.parametrize(
    "path_source, dir_target",
    [
        ("/test/path", "uploads"),
        ("/data/models/model.pkl", "models"),
        ("/logs/experiment.log", "logs"),
        ("/artifacts/plot.png", "plots"),
        ("/results/summary.json", "results"),
    ],
)
def test_upload(
    adapter_factory: Callable[[Optional[str], Optional[str]], InMemoryRunAdapter],
    path_source: str,
    dir_target: str,
):
    adapter: InMemoryRunAdapter = adapter_factory(None, None)
    assert len(adapter.uploaded_files) == 0
    result = adapter.upload(path_source=path_source, dir_target=dir_target)
    assert result is None
    assert len(adapter.uploaded_files) == 1
    uploaded_entry = adapter.uploaded_files[0]
    expected_entry = {"path_source": path_source, "dir_target": dir_target}
    assert uploaded_entry == expected_entry
