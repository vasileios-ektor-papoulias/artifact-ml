import os
import tempfile
from typing import Callable, Dict, List, Optional, Tuple
from unittest.mock import ANY, MagicMock

import pytest
from artifact_experiment._impl.backends.mlflow.adapter import MlflowRunAdapter
from artifact_experiment._impl.backends.mlflow.loggers.plot_collections import (
    MlflowPlotCollectionLogger,
)
from artifact_experiment._utils.incremental_path_generator import IncrementalPathGenerator
from pytest_mock import MockerFixture


@pytest.mark.unit
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
    mock_tempdir: Dict[str, MagicMock],
    plot_collection_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[MlflowRunAdapter, MlflowPlotCollectionLogger]
    ],
    experiment_id: str,
    run_id: str,
    ls_plot_collection_names: List[str],
    ls_plot_collections: List[Dict[str, Figure]],
):
    adapter, logger = plot_collection_logger_factory(experiment_id, run_id)
    ls_logged = []
    mock_get_ls_artifact_info = mocker.patch.object(
        adapter, "get_ls_artifact_info", return_value=ls_logged
    )
    mock_tempdir_cm = mock_tempdir["mock_tempdir_cm"]
    spy_format_path = mocker.spy(IncrementalPathGenerator, "format_path")
    mock_savefig = mocker.patch.object(Figure, "savefig")
    adapter_upload_spy = mocker.spy(adapter, "upload")
    for idx, (plot_collection_name, plot_collection) in enumerate(
        zip(ls_plot_collection_names, ls_plot_collections), start=1
    ):
        expected_get_call_count = idx
        expected_upload_call_count = len(ls_logged) + len(plot_collection)
        expected_tempdir_cm_call_count = idx
        expected_format_path_call_count = 2 * idx
        expected_savefig_call_count = len(ls_logged) + len(plot_collection)
        expected_next_idx = len(ls_logged) + 1
        expected_backend_plot_collection_path = os.path.join(
            "artifacts", "plot_collections", plot_collection_name
        )
        expected_tmp_dir = os.path.join("mock", "tmp", "dir")

        logger.log(artifact_name=plot_collection_name, artifact=plot_collection)
        assert mock_get_ls_artifact_info.call_count == expected_get_call_count
        mock_get_ls_artifact_info.assert_any_call(
            backend_path=expected_backend_plot_collection_path
        )
        assert mock_tempdir_cm.call_count == expected_tempdir_cm_call_count
        assert spy_format_path.call_count == expected_format_path_call_count
        spy_format_path.assert_any_call(
            dir_path=expected_backend_plot_collection_path, next_idx=expected_next_idx
        )
        spy_format_path.assert_any_call(dir_path=expected_tmp_dir, next_idx=expected_next_idx)
        assert mock_savefig.call_count == expected_savefig_call_count
        assert adapter_upload_spy.call_count == expected_upload_call_count
        for plot_name, plot in plot_collection.items():
            expected_backend_plot_path = os.path.join(
                expected_backend_plot_collection_path, f"{expected_next_idx}"
            )
            expected_path_source = os.path.join(
                expected_tmp_dir, f"{expected_next_idx}", f"{plot_name}.png"
            )
            mock_savefig.assert_any_call(
                fname=expected_path_source, format="png", bbox_inches="tight"
            )
            adapter_upload_spy.assert_any_call(
                path_source=expected_path_source, dir_target=expected_backend_plot_path
            )
            ls_logged.append(plot)


@pytest.mark.integration
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
def test_log_fs_integration(
    mocker: MockerFixture,
    plot_collection_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[MlflowRunAdapter, MlflowPlotCollectionLogger]
    ],
    experiment_id: str,
    run_id: str,
    ls_plot_collection_names: List[str],
    ls_plot_collections: List[Dict[str, Figure]],
):
    adapter, logger = plot_collection_logger_factory(experiment_id, run_id)
    ls_logged = []
    mock_get_ls_artifact_info = mocker.patch.object(
        adapter, "get_ls_artifact_info", return_value=ls_logged
    )
    spy_tempdir = mocker.spy(tempfile, "TemporaryDirectory")
    spy_format_path = mocker.spy(IncrementalPathGenerator, "format_path")
    spy_savefig = mocker.spy(Figure, "savefig")
    adapter_upload_spy = mocker.spy(adapter, "upload")
    for idx, (plot_collection_name, plot_collection) in enumerate(
        zip(ls_plot_collection_names, ls_plot_collections), start=1
    ):
        expected_get_call_count = idx
        expected_upload_call_count = len(ls_logged) + len(plot_collection)
        expected_tempdir_cm_call_count = idx
        expected_format_path_call_count = 2 * idx
        expected_savefig_call_count = len(ls_logged) + len(plot_collection)
        expected_next_idx = len(ls_logged) + 1
        expected_backend_plot_collection_path = os.path.join(
            "artifacts", "plot_collections", plot_collection_name
        )

        logger.log(artifact_name=plot_collection_name, artifact=plot_collection)
        expected_tmp_dir = spy_tempdir.spy_return.name
        assert mock_get_ls_artifact_info.call_count == expected_get_call_count
        mock_get_ls_artifact_info.assert_any_call(
            backend_path=expected_backend_plot_collection_path
        )
        assert spy_tempdir.call_count == expected_tempdir_cm_call_count
        assert spy_format_path.call_count == expected_format_path_call_count
        spy_format_path.assert_any_call(
            dir_path=expected_backend_plot_collection_path, next_idx=expected_next_idx
        )
        spy_format_path.assert_any_call(dir_path=expected_tmp_dir, next_idx=expected_next_idx)
        assert spy_savefig.call_count == expected_savefig_call_count
        assert adapter_upload_spy.call_count == expected_upload_call_count
        for plot_name, plot in plot_collection.items():
            expected_backend_plot_path = os.path.join(
                expected_backend_plot_collection_path, f"{expected_next_idx}"
            )
            expected_path_source = os.path.join(
                expected_tmp_dir, f"{expected_next_idx}", f"{plot_name}.png"
            )
            spy_savefig.assert_any_call(
                ANY, fname=expected_path_source, format="png", bbox_inches="tight"
            )
            adapter_upload_spy.assert_any_call(
                path_source=expected_path_source, dir_target=expected_backend_plot_path
            )
            ls_logged.append(plot)
