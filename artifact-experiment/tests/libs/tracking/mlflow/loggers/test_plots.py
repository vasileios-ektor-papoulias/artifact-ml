import os
import tempfile
from typing import Callable, Dict, List, Optional, Tuple
from unittest.mock import ANY, MagicMock

import pytest
from artifact_experiment._impl.mlflow.adapter import MlflowRunAdapter
from artifact_experiment._impl.mlflow.loggers.plots import MlflowPlotLogger
from artifact_experiment._utils.incremental_path_generator import IncrementalPathGenerator
from pytest_mock import MockerFixture


@pytest.mark.unit
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
    mock_tempdir: Dict[str, MagicMock],
    plot_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[MlflowRunAdapter, MlflowPlotLogger]
    ],
    experiment_id: str,
    run_id: str,
    ls_plot_names: List[str],
    ls_plots: List[Figure],
):
    adapter, logger = plot_logger_factory(experiment_id, run_id)
    ls_logged = []
    mock_get_ls_artifact_info = mocker.patch.object(
        adapter, "get_ls_artifact_info", return_value=ls_logged
    )
    mock_tempdir_cm = mock_tempdir["mock_tempdir_cm"]
    spy_format_path = mocker.spy(IncrementalPathGenerator, "format_path")
    mock_savefig = mocker.patch.object(Figure, "savefig")
    adapter_upload_spy = mocker.spy(adapter, "upload")
    for idx, (plot_name, plot) in enumerate(zip(ls_plot_names, ls_plots), start=1):
        expected_get_call_count = idx
        expected_log_call_count = idx
        expected_tempdir_cm_call_count = idx
        expected_format_path_call_count = idx
        expected_savefig_call_count = idx
        expected_next_idx = len(ls_logged) + 1
        expected_backend_path = os.path.join("artifacts", "plots", plot_name)
        expected_tmp_dir = os.path.join("mock", "tmp", "dir")
        expected_path_source = os.path.join(expected_tmp_dir, f"{expected_next_idx}.png")
        logger.log(artifact_name=plot_name, artifact=plot)
        assert mock_get_ls_artifact_info.call_count == expected_get_call_count
        mock_get_ls_artifact_info.assert_any_call(backend_path=expected_backend_path)
        assert mock_tempdir_cm.call_count == expected_tempdir_cm_call_count
        assert spy_format_path.call_count == expected_format_path_call_count
        spy_format_path.assert_any_call(
            dir_path=expected_tmp_dir, next_idx=expected_next_idx, fmt="png"
        )
        assert mock_savefig.call_count == expected_savefig_call_count
        mock_savefig.assert_any_call(fname=expected_path_source, format="png", bbox_inches="tight")
        assert adapter_upload_spy.call_count == expected_log_call_count
        adapter_upload_spy.assert_any_call(
            path_source=expected_path_source, dir_target=expected_backend_path
        )
        ls_logged.append(plot)


@pytest.mark.integration
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
def test_log_fs_integration(
    mocker: MockerFixture,
    plot_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[MlflowRunAdapter, MlflowPlotLogger]
    ],
    experiment_id: str,
    run_id: str,
    ls_plot_names: List[str],
    ls_plots: List[Figure],
):
    adapter, logger = plot_logger_factory(experiment_id, run_id)
    ls_logged = []
    mock_get_ls_artifact_info = mocker.patch.object(
        adapter, "get_ls_artifact_info", return_value=ls_logged
    )
    spy_tempdir = mocker.spy(tempfile, "TemporaryDirectory")
    spy_savefig = mocker.spy(Figure, "savefig")
    adapter_upload_spy = mocker.spy(adapter, "upload")
    for idx, (plot_name, plot) in enumerate(zip(ls_plot_names, ls_plots), start=1):
        expected_get_call_count = idx
        expected_log_call_count = idx
        expected_tempdir_call_count = idx
        expected_savefig_call_count = idx
        expected_backend_path = os.path.join("artifacts", "plots", plot_name)
        logger.log(artifact_name=plot_name, artifact=plot)
        expected_path_source = os.path.join(
            spy_tempdir.spy_return.name, f"{1 + len(ls_logged)}.png"
        )
        assert mock_get_ls_artifact_info.call_count == expected_get_call_count
        mock_get_ls_artifact_info.assert_any_call(backend_path=expected_backend_path)
        assert spy_tempdir.call_count == expected_tempdir_call_count
        assert spy_savefig.call_count == expected_savefig_call_count
        spy_savefig.assert_any_call(
            ANY, fname=expected_path_source, format="png", bbox_inches="tight"
        )
        assert adapter_upload_spy.call_count == expected_log_call_count
        adapter_upload_spy.assert_any_call(
            path_source=expected_path_source, dir_target=expected_backend_path
        )
        ls_logged.append(plot)
