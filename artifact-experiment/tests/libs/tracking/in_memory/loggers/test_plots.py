from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pytest
from artifact_experiment._impl.backends.in_memory.adapter import InMemoryRunAdapter
from artifact_experiment._impl.backends.in_memory.loggers.plots import InMemoryPlotLogger
from pytest_mock import MockerFixture


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_plot_names, ls_plots, ls_step",
    [
        ("exp1", "run1", [], [], []),
        ("exp1", "run1", ["plot_1"], ["plot_1"], [1]),
        (
            "exp1",
            "run1",
            ["plot_1", "plot_2"],
            ["plot_1", "plot_2"],
            [1, 1],
        ),
        (
            "exp1",
            "run1",
            ["plot_1", "plot_3"],
            ["plot_1", "plot_3"],
            [1, 1],
        ),
        (
            "exp1",
            "run1",
            ["plot_1", "plot_2", "plot_3"],
            ["plot_1", "plot_2", "plot_3"],
            [1, 1, 1],
        ),
        (
            "exp1",
            "run1",
            [
                "plot_1",
                "plot_2",
                "plot_3",
                "plot_4",
                "plot_5",
            ],
            [
                "plot_1",
                "plot_2",
                "plot_3",
                "plot_4",
                "plot_5",
            ],
            [1, 1, 1, 1, 1],
        ),
        (
            "exp1",
            "run1",
            ["plot_1", "plot_2", "plot_1"],
            ["plot_1", "plot_2", "plot_3"],
            [1, 1, 2],
        ),
        (
            "exp1",
            "run1",
            [
                "plot_1",
                "plot_1",
                "plot_2",
                "plot_2",
                "plot_2",
                "plot_1",
                "plot_3",
            ],
            [
                "plot_1",
                "plot_2",
                "plot_3",
                "plot_1",
                "plot_2",
                "plot_3",
                "plot_5",
            ],
            [1, 2, 1, 2, 3, 3, 1],
        ),
    ],
    indirect=["ls_plots"],
)
def test_log(
    mocker: MockerFixture,
    plot_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[InMemoryRunAdapter, InMemoryPlotLogger]
    ],
    experiment_id: str,
    run_id: str,
    ls_plot_names: List[str],
    ls_plots: List[Figure],
    ls_step: List[int],
):
    adapter, logger = plot_logger_factory(experiment_id, run_id)
    spy = mocker.spy(adapter, "log_plot")
    for name, plot in zip(ls_plot_names, ls_plots):
        logger.log(artifact_name=name, artifact=plot)
    assert spy.call_count == len(ls_plots)
    for idx, call_args in enumerate(spy.call_args_list):
        name = ls_plot_names[idx]
        plot = ls_plots[idx]
        step = ls_step[idx]
        expected_path = f"{experiment_id}/{run_id}/plots/{name}/{step}"
        actual_path = call_args.kwargs["path"]
        assert Path(actual_path) == Path(expected_path)
        assert call_args.kwargs["plot"] is plot
