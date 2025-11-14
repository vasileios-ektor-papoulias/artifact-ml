from typing import Callable, Optional, Tuple
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from artifact_experiment._impl.mlflow.adapter import MlflowNativeRun, MlflowRunAdapter
from mlflow.entities import RunStatus
from pytest_mock import MockerFixture


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
    adapter = MlflowRunAdapter.build(experiment_id=experiment_id, run_id=run_id)
    mock_get_env.assert_called_once_with(env_var_name="MLFLOW_TRACKING_URI")
    mock_mlflow_client_constructor.assert_called_once_with(tracking_uri="mock-tracking_uri")
    expected_experiment_uuid = f"{adapter.experiment_id}_uuid"
    expected_run_uuid = f"{adapter.run_id}_uuid"
    assert adapter.is_active is True
    assert adapter.experiment_id == experiment_id
    assert adapter.experiment_uuid == expected_experiment_uuid
    if run_id is not None:
        assert adapter.run_id == run_id
    else:
        assert adapter.run_id is not None
        assert len(adapter.run_id) == standard_uuid_length
        UUID(adapter.run_id)
    assert adapter.run_uuid == expected_run_uuid


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id",
    [
        ("exp1", "run1"),
        ("exp1", None),
    ],
)
def test_from_native_run(
    native_run_factory: Callable[
        [Optional[str], Optional[str]], Tuple[MagicMock, MagicMock, MagicMock, MlflowNativeRun]
    ],
    experiment_id: str,
    run_id: Optional[str],
):
    _, _, _, native_run = native_run_factory(experiment_id, run_id)
    adapter = MlflowRunAdapter.from_native_run(native_run=native_run)
    assert adapter.is_active is True
    assert adapter.experiment_id == native_run.experiment.name
    assert adapter.experiment_uuid == native_run.experiment.experiment_id
    assert adapter.experiment_id == experiment_id
    assert adapter.run_id == native_run.run.info.run_name
    assert adapter.run_uuid == native_run.run.info.run_id
    if run_id is not None:
        assert adapter.run_id == run_id
        assert native_run.run.info.run_name == run_id


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id",
    [
        ("exp1", "run1"),
        ("exp1", None),
    ],
)
def test_native_context_manager(
    adapter_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[MagicMock, MagicMock, MagicMock, MlflowNativeRun, MlflowRunAdapter],
    ],
    experiment_id: str,
    run_id: Optional[str],
):
    _, _, _, native_run, adapter = adapter_factory(experiment_id, run_id)
    with adapter.native() as ctx_native_run:
        assert ctx_native_run is native_run


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id",
    [
        ("exp1", "run1"),
        ("exp1", None),
    ],
)
def test_stop_run(
    adapter_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[MagicMock, MagicMock, MagicMock, MlflowNativeRun, MlflowRunAdapter],
    ],
    experiment_id: str,
    run_id: Optional[str],
):
    mock_client, _, mock_run, _, adapter = adapter_factory(experiment_id, run_id)
    assert adapter.is_active
    assert adapter.run_status == RunStatus.to_string(RunStatus.RUNNING)
    adapter.stop()
    mock_client.set_terminated.assert_called_once_with(run_id=mock_run.info.run_id)
    assert not adapter.is_active
    assert adapter.run_status == RunStatus.to_string(RunStatus.FINISHED)


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, path_source, dir_target, expected_artifact_path",
    [
        ("exp1", "run1", "/test/path", "uploads", "artifact_ml/uploads"),
        ("exp1", None, "/data/models/model.pkl", "models", "artifact_ml/models"),
        ("exp1", "run1", "/logs/experiment.log", "logs", "artifact_ml/logs"),
        ("exp1", None, "/artifacts/plot.png", "plots", "artifact_ml/plots"),
        ("exp1", "run1", "/results/summary.json", "results", "artifact_ml/results"),
    ],
)
def test_upload(
    mocker: MockerFixture,
    adapter_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[MagicMock, MagicMock, MagicMock, MlflowNativeRun, MlflowRunAdapter],
    ],
    experiment_id: str,
    run_id: Optional[str],
    path_source: str,
    dir_target: str,
    expected_artifact_path: str,
):
    mock_client, _, mock_run, _, adapter = adapter_factory(experiment_id, run_id)
    mock_log_artifact = mocker.patch.object(mock_client, "log_artifact")
    adapter.upload(path_source=path_source, dir_target=dir_target)
    mock_log_artifact.assert_called_once_with(
        run_id=mock_run.info.run_id,
        local_path=path_source,
        artifact_path=expected_artifact_path,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, backend_path, artifact_result, step, expected_key",
    [
        ("exp1", "run1", "/test/path", "score_1", 1, "artifact_ml/test/path"),
        ("exp1", None, "/test/path", "score_2", 2, "artifact_ml/test/path"),
        ("exp1", "run1", "/test/path", "array_1", 1, "artifact_ml/test/path"),
        ("exp1", None, "/test/path", "array_2", 3, "artifact_ml/test/path"),
        ("exp1", "run1", "/test/path", "plot_1", 1, "artifact_ml/test/path"),
        ("exp1", None, "/test/path", "plot_2", 4, "artifact_ml/test/path"),
        ("exp1", "run1", "/test/path", "score_collection_1", 1, "artifact_ml/test/path"),
        ("exp1", None, "/test/path", "score_collection_2", 1, "artifact_ml/test/path"),
        ("exp1", "run1", "/test/path", "array_collection_1", 1, "artifact_ml/test/path"),
        ("exp1", None, "/test/path", "array_collection_2", 8, "artifact_ml/test/path"),
        ("exp1", "run1", "/test/path", "plot_collection_1", 1, "artifact_ml/test/path"),
        ("exp1", None, "/test/path", "plot_collection_2", 11, "artifact_ml/test/path"),
    ],
    indirect=["artifact_result"],
)
def test_log_score(
    mocker: MockerFixture,
    adapter_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[MagicMock, MagicMock, MagicMock, MlflowNativeRun, MlflowRunAdapter],
    ],
    experiment_id: str,
    run_id: str,
    backend_path: str,
    artifact_result: float,
    step: int,
    expected_key: str,
):
    mock_client, _, mock_run, _, adapter = adapter_factory(experiment_id, run_id)
    mock_log_metric = mocker.patch.object(mock_client, "log_metric")
    adapter.log_score(backend_path=backend_path, value=artifact_result, step=step)
    mock_log_metric.assert_called_once_with(
        run_id=mock_run.info.run_id, key=expected_key, value=artifact_result, step=step
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, backend_path, expected_key",
    [
        ("exp1", "run1", "/artifacts/checkpoints", "artifact_ml/artifacts/checkpoints"),
        ("exp1", None, "/models", "artifact_ml/models"),
    ],
)
def test_get_ls_artifact_info(
    mocker: MockerFixture,
    adapter_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[MagicMock, MagicMock, MagicMock, MlflowNativeRun, MlflowRunAdapter],
    ],
    experiment_id: str,
    run_id: Optional[str],
    backend_path: str,
    expected_key: str,
):
    mock_client, _, mock_run, _, adapter = adapter_factory(experiment_id, run_id)
    mock_list_artifacts = mocker.patch.object(mock_client, "list_artifacts", return_value=[])
    adapter.get_ls_artifact_info(backend_path=backend_path)
    mock_list_artifacts.assert_called_once_with(run_id=mock_run.info.run_id, path=expected_key)


@pytest.mark.unit
@pytest.mark.parametrize(
    "experiment_id, run_id, backend_path, expected_key",
    [
        ("exp1", "run1", "/scores/val", "artifact_ml/scores/val"),
        ("exp1", None, "/metrics/loss", "artifact_ml/metrics/loss"),
    ],
)
def test_get_ls_score_history(
    mocker: MockerFixture,
    adapter_factory: Callable[
        [Optional[str], Optional[str]],
        Tuple[MagicMock, MagicMock, MagicMock, MlflowNativeRun, MlflowRunAdapter],
    ],
    experiment_id: str,
    run_id: Optional[str],
    backend_path: str,
    expected_key: str,
):
    mock_client, _, mock_run, _, adapter = adapter_factory(experiment_id, run_id)
    mock_get_metric_history = mocker.patch.object(
        mock_client, "get_metric_history", return_value=[]
    )
    adapter.get_ls_score_history(backend_path=backend_path)
    mock_get_metric_history.assert_called_once_with(run_id=mock_run.info.run_id, key=expected_key)
