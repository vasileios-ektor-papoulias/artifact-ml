import os
from typing import Callable, List, Optional, Tuple
from unittest.mock import ANY

import pytest
from artifact_experiment.libs.tracking.filesystem.adapter import FilesystemRunAdapter
from artifact_experiment.libs.tracking.filesystem.loggers.plots import FilesystemPlotLogger
from matplotlib.figure import Figure
from pytest_mock import MockerFixture


@pytest.fixture
def patched_incremental_generator(mocker: MockerFixture) -> List[str]:
    generated_paths: List[str] = []

    def fake_generate(dir_path: str, fmt: str) -> str:
        count = 1 + sum(1 for p in generated_paths if p.startswith(dir_path + os.sep))
        path = os.path.join(dir_path, f"{count}.{fmt}")
        generated_paths.append(path)
        return path

    mocker.patch(
        "artifact_experiment.libs.utils.incremental_path_generator.IncrementalPathGenerator.generate",
        side_effect=fake_generate,
    )

    return generated_paths


@pytest.mark.parametrize(
    "experiment_id, run_id, ls_plot_names, ls_plots, ls_step",
    [
        ("exp1", "run1", [], [], []),
        ("exp1", "run1", ["plot_1"], ["plot_1"], [1]),
        ("exp1", "run1", ["plot_1", "plot_2"], ["plot_1", "plot_2"], [1, 1]),
        ("exp1", "run1", ["plot_1", "plot_1"], ["plot_1", "plot_2"], [1, 2]),
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

    ls_savefig_mocks = []
    for plot in ls_plots:
        mock = mocker.patch.object(plot, "savefig")
        ls_savefig_mocks.append((plot, mock))

    for name, plot in zip(ls_plot_names, ls_plots):
        logger.log(artifact_name=name, artifact=plot)

    assert len(patched_incremental_generator) == len(ls_plots)

    for i, (name, plot, step) in enumerate(zip(ls_plot_names, ls_plots, ls_step)):
        expected_path = os.path.join(
            "test_root", experiment_id, run_id, "artifacts", "plots", name, f"{step}.png"
        )
        assert patched_incremental_generator[i] == expected_path
        _, savefig_mock = ls_savefig_mocks[i]
        savefig_mock.assert_called_with(
            fname=expected_path,
            dpi=ANY,
            bbox_inches=ANY,
            format=ANY,
        )
