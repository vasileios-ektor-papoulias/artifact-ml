from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pytest
from artifact_experiment.libs.tracking.in_memory.adapter import InMemoryRunAdapter
from artifact_experiment.libs.tracking.in_memory.loggers.plot_collections import (
    InMemoryPlotCollectionLogger,
)
from pytest_mock import MockerFixture


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_plot_collection_names, ls_plot_collections, ls_step",
    [
        ("exp1", "run1", [], [], []),
        ("exp1", "run1", ["array_collection_1"], ["array_collection_1"], [1]),
        (
            "exp1",
            "run1",
            ["array_collection_1", "array_collection_2"],
            ["array_collection_1", "array_collection_2"],
            [1, 1],
        ),
        (
            "exp1",
            "run1",
            ["array_collection_1", "array_collection_3"],
            ["array_collection_1", "array_collection_3"],
            [1, 1],
        ),
        (
            "exp1",
            "run1",
            ["array_collection_1", "array_collection_2", "array_collection_3"],
            ["array_collection_1", "array_collection_2", "array_collection_3"],
            [1, 1, 1],
        ),
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
            [
                "array_collection_1",
                "array_collection_2",
                "array_collection_3",
                "array_collection_4",
                "array_collection_5",
            ],
            [1, 1, 1, 1, 1],
        ),
        (
            "exp1",
            "run1",
            ["array_collection_1", "array_collection_2", "array_collection_1"],
            ["array_collection_1", "array_collection_2", "array_collection_3"],
            [1, 1, 2],
        ),
        (
            "exp1",
            "run1",
            [
                "array_collection_1",
                "array_collection_1",
                "array_collection_2",
                "array_collection_2",
                "array_collection_2",
                "array_collection_1",
                "array_collection_3",
            ],
            [
                "array_collection_1",
                "array_collection_2",
                "array_collection_3",
                "array_collection_1",
                "array_collection_2",
                "array_collection_3",
                "array_collection_5",
            ],
            [1, 2, 1, 2, 3, 3, 1],
        ),
    ],
    indirect=["ls_plot_collections"],
)
def test_log(
    mocker: MockerFixture,
    plot_collection_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRunAdapter, InMemoryPlotCollectionLogger]
    ],
    experiment_id: str,
    run_id: str,
    ls_plot_collection_names: List[str],
    ls_plot_collections: List[Dict[str, Figure]],
    ls_step: List[int],
):
    adapter, logger = plot_collection_logger_factory(experiment_id, run_id)
    spy = mocker.spy(adapter, "log_plot_collection")
    for name, plot_collection in zip(ls_plot_collection_names, ls_plot_collections):
        logger.log(artifact_name=name, artifact=plot_collection)
    assert spy.call_count == len(ls_plot_collections)
    for idx, call_args in enumerate(spy.call_args_list):
        name = ls_plot_collection_names[idx]
        plot_collection = ls_plot_collections[idx]
        step = ls_step[idx]
        expected_path = f"{experiment_id}/{run_id}/plot_collections/{name}/{step}"
        actual_path = call_args.kwargs["path"]
        assert Path(actual_path) == Path(expected_path)
        assert call_args.kwargs["plot_collection"] is plot_collection
