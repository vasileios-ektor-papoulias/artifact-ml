import os
from typing import Callable, List, Optional, Tuple

import pytest
from artifact_experiment.libs.tracking.neptune.adapter import NeptuneRunAdapter
from artifact_experiment.libs.tracking.neptune.loggers.plots import NeptunePlotLogger
from matplotlib.figure import Figure
from pytest_mock import MockerFixture


@pytest.mark.parametrize(
    "experiment_id, run_id, ls_plot_names, ls_plots",
    [
        ("exp1", "run1", [], []),
        ("exp1", "run1", ["plot_1"], ["plot_1"]),
        ("exp1", "run1", ["plot_1", "plot_2"], ["plot_1", "plot_2"]),
        ("exp1", "run1", ["plot_1", "plot_3"], ["plot_1", "plot_3"]),
        ("exp1", "run1", ["plot_1", "plot_2", "plot_3"], ["plot_1", "plot_2", "plot_3"]),
        (
            "exp1",
            "run1",
            ["plot_1", "plot_2", "plot_3", "plot_4", "plot_5"],
            ["plot_1", "plot_2", "plot_3", "plot_4", "plot_5"],
        ),
    ],
    indirect=["ls_plots"],
)
def test_log(
    mocker: MockerFixture,
    plot_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[NeptuneRunAdapter, NeptunePlotLogger]
    ],
    experiment_id: str,
    run_id: str,
    ls_plot_names: List[str],
    ls_plots: List[Figure],
):
    adapter, logger = plot_logger_factory(experiment_id, run_id)
    spy_adapter_log = mocker.spy(adapter, "log")
    for plot_name, plot in zip(ls_plot_names, ls_plots):
        logger.log(artifact_name=plot_name, artifact=plot)
    assert spy_adapter_log.call_count == len(ls_plots)
    for idx, call_args in enumerate(spy_adapter_log.call_args_list):
        plot_name = ls_plot_names[idx]
        plot = ls_plots[idx]
        expected_path = os.path.join("artifacts", "plots", plot_name)
        assert call_args.kwargs == {"artifact_path": expected_path, "artifact": plot}
