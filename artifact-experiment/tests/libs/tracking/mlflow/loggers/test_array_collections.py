import os
import tempfile
from typing import Callable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest
from artifact_experiment.libs.tracking.mlflow.adapter import MlflowRunAdapter
from artifact_experiment.libs.tracking.mlflow.loggers.array_collections import (
    MlflowArrayCollectionLogger,
)
from artifact_experiment.libs.utils.incremental_path_generator import IncrementalPathGenerator
from pytest_mock import MockerFixture


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_array_collection_names, ls_array_collections",
    [
        ("exp1", "run1", [], []),
        ("exp1", "run1", ["array_collection_1"], ["array_collection_1"]),
        (
            "exp1",
            "run1",
            ["array_collection_1", "array_collection_2"],
            ["array_collection_1", "array_collection_2"],
        ),
        (
            "exp1",
            "run1",
            ["array_collection_1", "array_collection_3"],
            ["array_collection_1", "array_collection_3"],
        ),
        (
            "exp1",
            "run1",
            ["array_collection_1", "array_collection_2", "array_collection_3"],
            ["array_collection_1", "array_collection_2", "array_collection_3"],
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
        ),
    ],
    indirect=["ls_array_collections"],
)
def test_log(
    mocker: MockerFixture,
    mock_tempdir: Dict[str, MagicMock],
    array_collection_logger_factory: Callable[
        [Optional[str], Optional[str]], Tuple[MlflowRunAdapter, MlflowArrayCollectionLogger]
    ],
    experiment_id: str,
    run_id: str,
    ls_array_collection_names: List[str],
    ls_array_collections: List[Dict[str, np.ndarray]],
):
    adapter, logger = array_collection_logger_factory(experiment_id, run_id)
    ls_logged = []
    mock_get_ls_artifact_info = mocker.patch.object(
        adapter, "get_ls_artifact_info", return_value=ls_logged
    )
    mock_tempdir_cm = mock_tempdir["mock_tempdir_cm"]
    spy_format_path = mocker.spy(IncrementalPathGenerator, "format_path")
    mock_np_savez = mocker.patch.object(np, "savez_compressed")
    adapter_upload_spy = mocker.spy(adapter, "upload")
    for idx, (array_collection_name, array_collection) in enumerate(
        zip(ls_array_collection_names, ls_array_collections), start=1
    ):
        expected_get_call_count = idx
        expected_log_call_count = idx
        expected_tempdir_cm_call_count = idx
        expected_format_path_call_count = idx
        expected_np_save_call_count = idx
        expected_next_idx = len(ls_logged) + 1
        expected_backend_path = os.path.join(
            "artifacts", "array_collections", array_collection_name
        )
        expected_tmp_dir = os.path.join("mock", "tmp", "dir")
        expected_path_source = os.path.join(expected_tmp_dir, f"{expected_next_idx}.npz")
        logger.log(artifact_name=array_collection_name, artifact=array_collection)
        assert mock_get_ls_artifact_info.call_count == expected_get_call_count
        mock_get_ls_artifact_info.assert_any_call(backend_path=expected_backend_path)
        assert mock_tempdir_cm.call_count == expected_tempdir_cm_call_count
        assert spy_format_path.call_count == expected_format_path_call_count
        spy_format_path.assert_any_call(
            dir_path=expected_tmp_dir, next_idx=expected_next_idx, fmt="npz"
        )
        assert mock_np_savez.call_count == expected_np_save_call_count
        mock_np_savez.assert_any_call(
            file=expected_path_source,
            allow_pickle=True,
            **array_collection,
        )
        assert adapter_upload_spy.call_count == expected_log_call_count
        adapter_upload_spy.assert_any_call(
            path_source=expected_path_source, dir_target=expected_backend_path
        )
        ls_logged.append(array_collection)


@pytest.mark.integration
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_array_collection_names, ls_array_collections",
    [
        ("exp1", "run1", [], []),
        ("exp1", "run1", ["array_collection_1"], ["array_collection_1"]),
        (
            "exp1",
            "run1",
            ["array_collection_1", "array_collection_2"],
            ["array_collection_1", "array_collection_2"],
        ),
        (
            "exp1",
            "run1",
            ["array_collection_1", "array_collection_3"],
            ["array_collection_1", "array_collection_3"],
        ),
        (
            "exp1",
            "run1",
            ["array_collection_1", "array_collection_2", "array_collection_3"],
            ["array_collection_1", "array_collection_2", "array_collection_3"],
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
        ),
    ],
    indirect=["ls_array_collections"],
)
def test_log_fs_integration(
    mocker: MockerFixture,
    array_collection_logger_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[MlflowRunAdapter, MlflowArrayCollectionLogger],
    ],
    experiment_id: str,
    run_id: str,
    ls_array_collection_names: List[str],
    ls_array_collections: List[Dict[str, np.ndarray]],
):
    adapter, logger = array_collection_logger_factory(experiment_id, run_id)
    ls_logged = []
    mock_get_ls_artifact_info = mocker.patch.object(
        adapter, "get_ls_artifact_info", return_value=ls_logged
    )
    spy_tempdir = mocker.spy(tempfile, "TemporaryDirectory")
    spy_np_savez = mocker.spy(np, "savez_compressed")
    adapter_upload_spy = mocker.spy(adapter, "upload")
    for idx, (array_collection_name, array_collection) in enumerate(
        zip(ls_array_collection_names, ls_array_collections), start=1
    ):
        expected_get_call_count = idx
        expected_log_call_count = idx
        expected_tempdir_call_count = idx
        expected_np_save_call_count = idx
        expected_backend_path = os.path.join(
            "artifacts", "array_collections", array_collection_name
        )
        logger.log(artifact_name=array_collection_name, artifact=array_collection)
        expected_path_source = os.path.join(
            spy_tempdir.spy_return.name, f"{1 + len(ls_logged)}.npz"
        )
        assert mock_get_ls_artifact_info.call_count == expected_get_call_count
        mock_get_ls_artifact_info.assert_any_call(backend_path=expected_backend_path)
        assert spy_tempdir.call_count == expected_tempdir_call_count
        assert spy_np_savez.call_count == expected_np_save_call_count
        spy_np_savez.assert_any_call(
            file=expected_path_source,
            allow_pickle=True,
            **array_collection,
        )
        assert adapter_upload_spy.call_count == expected_log_call_count
        adapter_upload_spy.assert_any_call(
            path_source=expected_path_source, dir_target=expected_backend_path
        )
        ls_logged.append(array_collection)
