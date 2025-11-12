import os
from typing import Callable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from artifact_experiment.libs.tracking.mlflow.adapter import MlflowNativeRun, MlflowRunAdapter
from artifact_experiment.libs.tracking.mlflow.client import MlflowTrackingClient
from artifact_experiment.libs.tracking.mlflow.loggers.array_collections import (
    MlflowArrayCollectionLogger,
)
from artifact_experiment.libs.tracking.mlflow.loggers.arrays import MlflowArrayLogger
from artifact_experiment.libs.tracking.mlflow.loggers.plot_collections import (
    MlflowPlotCollectionLogger,
)
from artifact_experiment.libs.tracking.mlflow.loggers.plots import MlflowPlotLogger
from artifact_experiment.libs.tracking.mlflow.loggers.score_collections import (
    MlflowScoreCollectionLogger,
)
from artifact_experiment.libs.tracking.mlflow.loggers.scores import MlflowScoreLogger
from pytest_mock import MockerFixture


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id",
    [("exp1", "run1")],
)
def test_init(
    loggers_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[
            MlflowRunAdapter,
            MlflowScoreLogger,
            MlflowArrayLogger,
            MlflowPlotLogger,
            MlflowScoreCollectionLogger,
            MlflowArrayCollectionLogger,
            MlflowPlotCollectionLogger,
        ],
    ],
    experiment_id: str,
    run_id: str,
):
    (
        adapter,
        score_logger,
        array_logger,
        plot_logger,
        score_collection_logger,
        array_collection_logger,
        plot_collection_logger,
    ) = loggers_factory(experiment_id, run_id)
    client = MlflowTrackingClient(
        run=adapter,
        score_logger=score_logger,
        array_logger=array_logger,
        plot_logger=plot_logger,
        score_collection_logger=score_collection_logger,
        array_collection_logger=array_collection_logger,
        plot_collection_logger=plot_collection_logger,
    )
    assert isinstance(client, MlflowTrackingClient)
    assert client.run == adapter
    assert client.run.experiment_id == experiment_id
    assert client.run.run_id == run_id
    assert client.run.is_active is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id",
    [("exp1", "run1")],
)
def test_from_run(
    adapter_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[MagicMock, MagicMock, MagicMock, MlflowNativeRun, MlflowRunAdapter],
    ],
    experiment_id: str,
    run_id: str,
):
    _, _, _, _, adapter = adapter_factory(experiment_id, run_id)
    client = MlflowTrackingClient.from_run(run=adapter)
    assert isinstance(client, MlflowTrackingClient)
    assert client.run == adapter
    assert client.run.experiment_id == experiment_id
    assert client.run.run_id == run_id
    assert client.run.is_active is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id",
    [("exp1", "run1")],
)
def test_from_native_run(
    native_run_factory: Callable[
        [Optional[str], Optional[str]], Tuple[MagicMock, MagicMock, MagicMock, MlflowNativeRun]
    ],
    experiment_id: str,
    run_id: str,
):
    _, _, _, native_run = native_run_factory(experiment_id, run_id)
    client = MlflowTrackingClient.from_native_run(native_run=native_run)
    assert isinstance(client, MlflowTrackingClient)
    assert client.run.experiment_id == experiment_id
    assert client.run.run_id == run_id
    assert client.run.is_active is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id",
    [
        ("exp1", "run1"),
        ("exp1", None),
    ],
)
def test_build(
    reset_tracking_uri_cache,
    mock_get_env: MagicMock,
    mock_mlflow_client_constructor: MagicMock,
    standard_uuid_length: int,
    experiment_id: str,
    run_id: Optional[str],
):
    client = MlflowTrackingClient.build(experiment_id=experiment_id, run_id=run_id)
    mock_get_env.assert_called_once_with(env_var_name="MLFLOW_TRACKING_URI")
    mock_mlflow_client_constructor.assert_called_once_with(tracking_uri="mock-tracking_uri")
    assert isinstance(client, MlflowTrackingClient)
    assert client.run.is_active is True
    assert client.run.experiment_id == experiment_id
    if run_id is not None:
        assert client.run.run_id == run_id
    else:
        assert client.run.run_id is not None
        assert len(client.run.run_id) == standard_uuid_length
        UUID(client.run.run_id)


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_scores",
    [
        ("exp1", "run1", []),
        ("exp1", "run1", ["score_1"]),
        ("exp1", "run1", ["score_1", "score_2"]),
        ("exp1", "run1", ["score_1", "score_3"]),
        ("exp1", "run1", ["score_1", "score_2", "score_3"]),
        ("exp1", "run1", ["score_1", "score_2", "score_3", "score_4", "score_5"]),
    ],
    indirect=["ls_scores"],
)
def test_log_score(
    mocker: MockerFixture,
    client_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[
            MlflowRunAdapter,
            MlflowScoreLogger,
            MlflowArrayLogger,
            MlflowPlotLogger,
            MlflowScoreCollectionLogger,
            MlflowArrayCollectionLogger,
            MlflowPlotCollectionLogger,
            MlflowTrackingClient,
        ],
    ],
    experiment_id: str,
    run_id: str,
    ls_scores: List[float],
):
    (
        adapter,
        score_logger,
        _,
        _,
        _,
        _,
        _,
        client,
    ) = client_factory(experiment_id, run_id)
    ls_logged = []
    mock_get_ls_score_history = mocker.patch.object(
        adapter, "get_ls_score_history", return_value=ls_logged
    )
    spy_logger_log = mocker.spy(score_logger, "log")
    for idx, score in enumerate(ls_scores, start=1):
        score_name = f"score_{idx}"
        expected_backend_path = os.path.join("artifacts", "scores", score_name)
        expected_get_call_count = len(ls_logged) + 1
        expected_log_call_count = idx
        client.log_score(score=score, name=score_name)
        assert mock_get_ls_score_history.call_count == expected_get_call_count
        mock_get_ls_score_history.assert_any_call(backend_path=expected_backend_path)
        assert spy_logger_log.call_count == expected_log_call_count
        spy_logger_log.assert_any_call(
            artifact_name=score_name,
            artifact=score,
        )
        ls_logged.append(score)


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_arrays",
    [
        ("exp1", "run1", []),
        ("exp1", "run1", ["array_1"]),
        ("exp1", "run1", ["array_1", "array_2"]),
        ("exp1", "run1", ["array_1", "array_3"]),
        ("exp1", "run1", ["array_1", "array_2", "array_3"]),
        ("exp1", "run1", ["array_1", "array_2", "array_3", "array_4", "array_5"]),
    ],
    indirect=["ls_arrays"],
)
def test_log_array(
    mocker: MockerFixture,
    client_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[
            MlflowRunAdapter,
            MlflowScoreLogger,
            MlflowArrayLogger,
            MlflowPlotLogger,
            MlflowScoreCollectionLogger,
            MlflowArrayCollectionLogger,
            MlflowPlotCollectionLogger,
            MlflowTrackingClient,
        ],
    ],
    experiment_id: str,
    run_id: str,
    ls_arrays: List[Array],
):
    (
        adapter,
        _,
        array_logger,
        _,
        _,
        _,
        _,
        client,
    ) = client_factory(experiment_id, run_id)
    ls_logged = []
    mock_get_ls_artifact_info = mocker.patch.object(
        adapter, "get_ls_artifact_info", return_value=ls_logged
    )
    spy_logger_log = mocker.spy(array_logger, "log")
    for idx, array in enumerate(ls_arrays, start=1):
        array_name = f"array_{idx}"
        expected_backend_path = os.path.join("artifacts", "arrays", array_name)
        expected_get_call_count = len(ls_logged) + 1
        expected_log_call_count = idx
        client.log_array(array=array, name=array_name)
        assert mock_get_ls_artifact_info.call_count == expected_get_call_count
        mock_get_ls_artifact_info.assert_any_call(backend_path=expected_backend_path)
        assert spy_logger_log.call_count == expected_log_call_count
        spy_logger_log.assert_any_call(
            artifact_name=array_name,
            artifact=array,
        )
        ls_logged.append(array)


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_plots",
    [
        ("exp1", "run1", []),
        ("exp1", "run1", ["plot_1"]),
        ("exp1", "run1", ["plot_1", "plot_2"]),
        ("exp1", "run1", ["plot_1", "plot_3"]),
        ("exp1", "run1", ["plot_1", "plot_2", "plot_3"]),
        ("exp1", "run1", ["plot_1", "plot_2", "plot_3", "plot_4", "plot_5"]),
    ],
    indirect=["ls_plots"],
)
def test_log_plot(
    mocker: MockerFixture,
    client_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[
            MlflowRunAdapter,
            MlflowScoreLogger,
            MlflowArrayLogger,
            MlflowPlotLogger,
            MlflowScoreCollectionLogger,
            MlflowArrayCollectionLogger,
            MlflowPlotCollectionLogger,
            MlflowTrackingClient,
        ],
    ],
    experiment_id: str,
    run_id: str,
    ls_plots: List[Figure],
):
    (
        adapter,
        _,
        _,
        plot_logger,
        _,
        _,
        _,
        client,
    ) = client_factory(experiment_id, run_id)
    ls_logged = []
    mock_get_ls_artifact_info = mocker.patch.object(
        adapter, "get_ls_artifact_info", return_value=ls_logged
    )
    spy_logger_log = mocker.spy(plot_logger, "log")
    for idx, plot in enumerate(ls_plots, start=1):
        plot_name = f"plot_{idx}"
        expected_backend_path = os.path.join("artifacts", "plots", plot_name)
        expected_get_call_count = len(ls_logged) + 1
        expected_log_call_count = idx
        client.log_plot(plot=plot, name=plot_name)
        assert mock_get_ls_artifact_info.call_count == expected_get_call_count
        mock_get_ls_artifact_info.assert_any_call(backend_path=expected_backend_path)
        assert spy_logger_log.call_count == expected_log_call_count
        spy_logger_log.assert_any_call(
            artifact_name=plot_name,
            artifact=plot,
        )
        ls_logged.append(plot)


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_score_collections",
    [
        ("exp1", "run1", []),
        ("exp1", "run1", ["score_collection_1"]),
        ("exp1", "run1", ["score_collection_1", "score_collection_2"]),
        ("exp1", "run1", ["score_collection_1", "score_collection_3"]),
        ("exp1", "run1", ["score_collection_1", "score_collection_2", "score_collection_3"]),
        (
            "exp1",
            "run1",
            [
                "score_collection_1",
                "score_collection_2",
                "score_collection_3",
                "score_collection_4",
                "score_collection_5",
            ],
        ),
    ],
    indirect=["ls_score_collections"],
)
def test_log_score_collection(
    mocker: MockerFixture,
    client_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[
            MlflowRunAdapter,
            MlflowScoreLogger,
            MlflowArrayLogger,
            MlflowPlotLogger,
            MlflowScoreCollectionLogger,
            MlflowArrayCollectionLogger,
            MlflowPlotCollectionLogger,
            MlflowTrackingClient,
        ],
    ],
    experiment_id: str,
    run_id: str,
    ls_score_collections: List[Dict[str, float]],
):
    (
        adapter,
        _,
        _,
        _,
        score_collection_logger,
        _,
        _,
        client,
    ) = client_factory(experiment_id, run_id)
    ls_logged = []
    mock_get_ls_score_history = mocker.patch.object(
        adapter, "get_ls_score_history", return_value=ls_logged
    )
    spy_logger_log = mocker.spy(score_collection_logger, "log")
    for idx, score_collection in enumerate(ls_score_collections, start=1):
        score_collection_name = f"score_collection_{idx}"
        expected_score_collection_backend_path = os.path.join(
            "artifacts", "score_collections", score_collection_name
        )
        expected_get_call_count = len(ls_logged) + len(score_collection)
        expected_log_call_count = idx
        client.log_score_collection(score_collection=score_collection, name=score_collection_name)
        assert mock_get_ls_score_history.call_count == expected_get_call_count
        for score_name in score_collection.keys():
            expected_score_backend_path = os.path.join(
                expected_score_collection_backend_path, score_name
            )
            mock_get_ls_score_history.assert_any_call(backend_path=expected_score_backend_path)
        assert spy_logger_log.call_count == expected_log_call_count
        spy_logger_log.assert_any_call(
            artifact_name=score_collection_name,
            artifact=score_collection,
        )
        for score in score_collection.values():
            ls_logged.append(score)


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_array_collections",
    [
        ("exp1", "run1", []),
        ("exp1", "run1", ["array_collection_1"]),
        ("exp1", "run1", ["array_collection_1", "array_collection_2"]),
        ("exp1", "run1", ["array_collection_1", "array_collection_3"]),
        ("exp1", "run1", ["array_collection_1", "array_collection_2", "array_collection_3"]),
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
        ),
    ],
    indirect=["ls_array_collections"],
)
def test_log_array_collection(
    mocker: MockerFixture,
    client_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[
            MlflowRunAdapter,
            MlflowScoreLogger,
            MlflowArrayLogger,
            MlflowPlotLogger,
            MlflowScoreCollectionLogger,
            MlflowArrayCollectionLogger,
            MlflowPlotCollectionLogger,
            MlflowTrackingClient,
        ],
    ],
    experiment_id: str,
    run_id: str,
    ls_array_collections: List[Dict[str, Array]],
):
    (
        adapter,
        _,
        _,
        _,
        _,
        array_collection_logger,
        _,
        client,
    ) = client_factory(experiment_id, run_id)
    ls_logged = []
    mock_get_ls_artifact_info = mocker.patch.object(
        adapter, "get_ls_artifact_info", return_value=ls_logged
    )
    spy_logger_log = mocker.spy(array_collection_logger, "log")
    for idx, array_collection in enumerate(ls_array_collections, start=1):
        array_collection_name = f"array_collection_{idx}"
        expected_backend_path = os.path.join(
            "artifacts", "array_collections", array_collection_name
        )
        expected_get_call_count = len(ls_logged) + 1
        expected_log_call_count = idx
        client.log_array_collection(array_collection=array_collection, name=array_collection_name)
        assert mock_get_ls_artifact_info.call_count == expected_get_call_count

        mock_get_ls_artifact_info.assert_any_call(backend_path=expected_backend_path)
        assert spy_logger_log.call_count == expected_log_call_count
        spy_logger_log.assert_any_call(
            artifact_name=array_collection_name,
            artifact=array_collection,
        )
        ls_logged.append(array_collection)


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_plot_collections",
    [
        ("exp1", "run1", []),
        ("exp1", "run1", ["plot_collection_1"]),
        ("exp1", "run1", ["plot_collection_1", "plot_collection_2"]),
        ("exp1", "run1", ["plot_collection_1", "plot_collection_3"]),
        ("exp1", "run1", ["plot_collection_1", "plot_collection_2", "plot_collection_3"]),
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
        ),
    ],
    indirect=["ls_plot_collections"],
)
def test_log_plot_collection(
    mocker: MockerFixture,
    client_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[
            MlflowRunAdapter,
            MlflowScoreLogger,
            MlflowArrayLogger,
            MlflowPlotLogger,
            MlflowScoreCollectionLogger,
            MlflowArrayCollectionLogger,
            MlflowPlotCollectionLogger,
            MlflowTrackingClient,
        ],
    ],
    experiment_id: str,
    run_id: str,
    ls_plot_collections: List[Dict[str, Figure]],
):
    (
        adapter,
        _,
        _,
        _,
        _,
        _,
        plot_collection_logger,
        client,
    ) = client_factory(experiment_id, run_id)
    ls_logged = []
    mock_get_ls_artifact_info = mocker.patch.object(
        adapter, "get_ls_artifact_info", return_value=ls_logged
    )
    spy_logger_log = mocker.spy(plot_collection_logger, "log")
    for idx, plot_collection in enumerate(ls_plot_collections, start=1):
        plot_collection_name = f"plot_collection_{idx}"
        expected_backend_path = os.path.join("artifacts", "plot_collections", plot_collection_name)
        expected_get_call_count = len(ls_logged) + 1
        expected_log_call_count = idx
        client.log_plot_collection(plot_collection=plot_collection, name=plot_collection_name)
        assert mock_get_ls_artifact_info.call_count == expected_get_call_count
        mock_get_ls_artifact_info.assert_any_call(backend_path=expected_backend_path)
        assert spy_logger_log.call_count == expected_log_call_count
        spy_logger_log.assert_any_call(
            artifact_name=plot_collection_name,
            artifact=plot_collection,
        )
        ls_logged.append(plot_collection)


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, ls_file_entries",
    [
        ("exp1", "run1", []),
        ("exp1", "run1", [{"path_source": "/test/path1", "dir_target": "uploads"}]),
        (
            "exp1",
            "run1",
            [
                {"path_source": "/test/path1", "dir_target": "uploads"},
                {"path_source": "/another/path", "dir_target": "data"},
            ],
        ),
    ],
)
def test_upload(
    mocker: MockerFixture,
    client_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[
            MlflowRunAdapter,
            MlflowScoreLogger,
            MlflowArrayLogger,
            MlflowPlotLogger,
            MlflowScoreCollectionLogger,
            MlflowArrayCollectionLogger,
            MlflowPlotCollectionLogger,
            MlflowTrackingClient,
        ],
    ],
    experiment_id: str,
    run_id: str,
    ls_file_entries: List[Dict[str, str]],
):
    (
        adapter,
        _,
        _,
        _,
        _,
        _,
        _,
        client,
    ) = client_factory(experiment_id, run_id)
    spy_adapter_upload = mocker.spy(adapter, "upload")
    for file_entry in ls_file_entries:
        client.log_file(**file_entry)
    assert spy_adapter_upload.call_count == len(ls_file_entries)
    for entry in ls_file_entries:
        spy_adapter_upload.assert_any_call(
            path_source=entry["path_source"], dir_target=entry["dir_target"]
        )
