from typing import Callable, Dict, List, Optional, Tuple

import pytest
from artifact_experiment.libs.tracking.in_memory.adapter import InMemoryRunAdapter
from artifact_experiment.libs.tracking.in_memory.loggers.plot_collections import (
    InMemoryPlotCollectionLogger,
)
from matplotlib.figure import Figure
from pytest_mock import MockerFixture


@pytest.mark.parametrize(
    "experiment_id, run_id, ls_plot_collection_names, ls_plot_collections",
    [
        ("exp1", "run1", [], []),
        ("exp1", "run1", ["plot_collection_1"], ["plot_collection_1"]),
        (
            "exp1",
            "run1",
            ["plot_collection_1", "plot_collection_2"],
            ["plot_collection_1", "plot_collection_2"],
        ),
        (
            "exp1",
            "run1",
            ["plot_collection_1", "plot_collection_3"],
            ["plot_collection_1", "plot_collection_3"],
        ),
        (
            "exp1",
            "run1",
            ["plot_collection_1", "plot_collection_2", "plot_collection_3"],
            ["plot_collection_1", "plot_collection_2", "plot_collection_3"],
        ),
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
def test_log(
    mocker: MockerFixture,
    plot_collection_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRunAdapter, InMemoryPlotCollectionLogger]
    ],
    experiment_id: str,
    run_id: str,
    ls_plot_collection_names: List[str],
    ls_plot_collections: List[Dict[str, Figure]],
):
    adapter, logger = plot_collection_logger_factory(experiment_id, run_id)
    spy = mocker.spy(adapter, "log_plot_collection")
    for name, plot_collection in zip(ls_plot_collection_names, ls_plot_collections):
        logger.log(artifact_name=name, artifact=plot_collection)
    assert spy.call_count == len(ls_plot_collections)
    for idx, call_args in enumerate(spy.call_args_list):
        name = ls_plot_collection_names[idx]
        plot_collection = ls_plot_collections[idx]
        expected_path = f"{experiment_id}/{run_id}/plot_collections/{name}/1"
        assert call_args.kwargs == {"path": expected_path, "plot_collection": plot_collection}
