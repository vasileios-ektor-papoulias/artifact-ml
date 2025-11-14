import os
import tempfile
from typing import Callable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest
from artifact_experiment._impl.mlflow.adapter import MlflowRunAdapter
from artifact_experiment._impl.mlflow.loggers.arrays import MlflowArrayLogger
from artifact_experiment._utils.incremental_path_generator import IncrementalPathGenerator
from pytest_mock import MockerFixture


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_array_names, ls_arrays",
    [
        ("exp1", "run1", [], []),
        ("exp1", "run1", ["array_1"], ["array_1"]),
        ("exp1", "run1", ["array_1", "array_2"], ["array_1", "array_2"]),
        ("exp1", "run1", ["array_1", "array_3"], ["array_1", "array_3"]),
        ("exp1", "run1", ["array_1", "array_2", "array_3"], ["array_1", "array_2", "array_3"]),
        (
            "exp1",
            "run1",
            ["array_1", "array_2", "array_3", "array_4", "array_5"],
            ["array_1", "array_2", "array_3", "array_4", "array_5"],
        ),
    ],
    indirect=["ls_arrays"],
)
def test_log(
    mocker: MockerFixture,
    mock_tempdir: Dict[str, MagicMock],
    array_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[MlflowRunAdapter, MlflowArrayLogger]
    ],
    experiment_id: str,
    run_id: str,
    ls_array_names: List[str],
    ls_arrays: List[Array],
):
    adapter, logger = array_logger_factory(experiment_id, run_id)
    ls_logged = []
    mock_get_ls_artifact_info = mocker.patch.object(
        adapter, "get_ls_artifact_info", return_value=ls_logged
    )
    mock_tempdir_cm = mock_tempdir["mock_tempdir_cm"]
    spy_format_path = mocker.spy(IncrementalPathGenerator, "format_path")
    mock_np_save = mocker.patch.object(np, "save")
    adapter_upload_spy = mocker.spy(adapter, "upload")
    for idx, (array_name, array) in enumerate(zip(ls_array_names, ls_arrays), start=1):
        expected_get_call_count = idx
        expected_log_call_count = idx
        expected_tempdir_cm_call_count = idx
        expected_format_path_call_count = idx
        expected_np_save_call_count = idx
        expected_next_idx = len(ls_logged) + 1
        expected_backend_path = os.path.join("artifacts", "arrays", array_name)
        expected_tmp_dir = os.path.join("mock", "tmp", "dir")
        expected_path_source = os.path.join(expected_tmp_dir, f"{expected_next_idx}.npy")
        logger.log(artifact_name=array_name, artifact=array)
        assert mock_get_ls_artifact_info.call_count == expected_get_call_count
        mock_get_ls_artifact_info.assert_any_call(backend_path=expected_backend_path)
        assert mock_tempdir_cm.call_count == expected_tempdir_cm_call_count
        assert spy_format_path.call_count == expected_format_path_call_count
        spy_format_path.assert_any_call(
            dir_path=expected_tmp_dir, next_idx=expected_next_idx, fmt="npy"
        )
        assert mock_np_save.call_count == expected_np_save_call_count
        mock_np_save.assert_any_call(file=expected_path_source, arr=array)
        assert adapter_upload_spy.call_count == expected_log_call_count
        adapter_upload_spy.assert_any_call(
            path_source=expected_path_source, dir_target=expected_backend_path
        )
        ls_logged.append(array)


@pytest.mark.integration
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_array_names, ls_arrays",
    [
        ("exp1", "run1", [], []),
        ("exp1", "run1", ["array_1"], ["array_1"]),
        ("exp1", "run1", ["array_1", "array_2"], ["array_1", "array_2"]),
        ("exp1", "run1", ["array_1", "array_3"], ["array_1", "array_3"]),
        ("exp1", "run1", ["array_1", "array_2", "array_3"], ["array_1", "array_2", "array_3"]),
        (
            "exp1",
            "run1",
            ["array_1", "array_2", "array_3", "array_4", "array_5"],
            ["array_1", "array_2", "array_3", "array_4", "array_5"],
        ),
    ],
    indirect=["ls_arrays"],
)
def test_log_fs_integration(
    mocker: MockerFixture,
    array_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[MlflowRunAdapter, MlflowArrayLogger]
    ],
    experiment_id: str,
    run_id: str,
    ls_array_names: List[str],
    ls_arrays: List[Array],
):
    adapter, logger = array_logger_factory(experiment_id, run_id)
    ls_logged = []
    mock_get_ls_artifact_info = mocker.patch.object(
        adapter, "get_ls_artifact_info", return_value=ls_logged
    )
    spy_tempdir = mocker.spy(tempfile, "TemporaryDirectory")
    spy_np_save = mocker.spy(np, "save")
    adapter_upload_spy = mocker.spy(adapter, "upload")
    for idx, (array_name, array) in enumerate(zip(ls_array_names, ls_arrays), start=1):
        expected_get_call_count = idx
        expected_log_call_count = idx
        expected_tempdir_call_count = idx
        expected_np_save_call_count = idx
        expected_backend_path = os.path.join("artifacts", "arrays", array_name)
        logger.log(artifact_name=array_name, artifact=array)
        expected_path_source = os.path.join(
            spy_tempdir.spy_return.name, f"{1 + len(ls_logged)}.npy"
        )
        assert mock_get_ls_artifact_info.call_count == expected_get_call_count
        mock_get_ls_artifact_info.assert_any_call(backend_path=expected_backend_path)
        assert spy_tempdir.call_count == expected_tempdir_call_count
        assert spy_np_save.call_count == expected_np_save_call_count
        spy_np_save.assert_any_call(file=expected_path_source, arr=array)
        assert adapter_upload_spy.call_count == expected_log_call_count
        adapter_upload_spy.assert_any_call(
            path_source=expected_path_source, dir_target=expected_backend_path
        )
        ls_logged.append(array)
