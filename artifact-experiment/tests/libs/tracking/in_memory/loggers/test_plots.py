from typing import Callable, List, Optional, Tuple

import pytest
from artifact_experiment.libs.tracking.in_memory.adapter import InMemoryRunAdapter
from artifact_experiment.libs.tracking.in_memory.loggers.plots import InMemoryPlotLogger
from matplotlib.figure import Figure
from pytest_mock import MockerFixture


@pytest.mark.parametrize(
    "experiment_id, run_id, ls_plot_names, ls_plots",
    [
        ("exp1", "run1", [], []),
        ("exp1", "run1", ["plot_1"], ["plot_1"]),
        (
            "exp1",
            "run1",
            ["plot_1", "plot_2"],
            ["plot_1", "plot_2"],
        ),
        (
            "exp1",
            "run1",
            ["plot_1", "plot_3"],
            ["plot_1", "plot_3"],
        ),
        (
            "exp1",
            "run1",
            ["plot_1", "plot_2", "plot_3"],
            ["plot_1", "plot_2", "plot_3"],
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
):
    adapter, logger = plot_logger_factory(experiment_id, run_id)
    spy = mocker.spy(adapter, "log_plot")
    for name, plot in zip(ls_plot_names, ls_plots):
        logger.log(artifact_name=name, artifact=plot)
    assert spy.call_count == len(ls_plots)
    for idx, call_args in enumerate(spy.call_args_list):
        name = ls_plot_names[idx]
        plot = ls_plots[idx]
        expected_path = f"{experiment_id}/{run_id}/plots/{name}/1"
        assert call_args.kwargs == {"path": expected_path, "plot": plot}
