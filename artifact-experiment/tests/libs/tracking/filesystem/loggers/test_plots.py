import os
from typing import Callable, List, Optional, Tuple
from unittest.mock import ANY

import pytest
from artifact_experiment.libs.tracking.filesystem.adapter import FilesystemRunAdapter
from artifact_experiment.libs.tracking.filesystem.loggers.plots import FilesystemPlotLogger
from matplotlib.figure import Figure
from pytest_mock import MockerFixture


@pytest.mark.parametrize(
    "experiment_id, run_id, ls_plot_names, ls_plots, ls_step",
    [
        ("exp1", "run1", [], [], []),
        ("exp1", "run1", ["plot_1"], ["plot_1"], [1]),
        ("exp1", "run1", ["plot_1", "plot_2"], ["plot_1", "plot_2"], [1, 1]),
        ("exp1", "run1", ["plot_1", "plot_1"], ["plot_1", "plot_2"], [1, 2]),
        ("exp1", "run1", ["plot_1", "plot_1", "plot_1"], ["plot_1", "plot_2", "plot_1"], [1, 2, 3]),
    ],
    indirect=["ls_plots"],
)
def test_log_plot(
    mocker: MockerFixture,
    patched_incremental_generator: List[str],
    plot_logger_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[FilesystemRunAdapter, FilesystemPlotLogger],
    ],
    experiment_id: str,
    run_id: str,
    ls_plot_names: List[str],
    ls_plots: List[Figure],
    ls_step: List[int],
):
    _, logger = plot_logger_factory(experiment_id, run_id)
    savefig_mock = mocker.patch.object(Figure, "savefig")
    for name, plot in zip(ls_plot_names, ls_plots):
        logger.log(artifact_name=name, artifact=plot)
    assert len(patched_incremental_generator) == len(ls_plots)
    for i, (name, plot, step) in enumerate(zip(ls_plot_names, ls_plots, ls_step)):
        expected_path = os.path.join(
            "test_root", experiment_id, run_id, "artifacts", "plots", name, f"{step}.png"
        )
        assert patched_incremental_generator[i] == expected_path
        savefig_mock.assert_any_call(
            fname=expected_path,
            dpi=ANY,
            bbox_inches=ANY,
            format=ANY,
        )
