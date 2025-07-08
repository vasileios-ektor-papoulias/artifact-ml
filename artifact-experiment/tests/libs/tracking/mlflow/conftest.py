from typing import Callable, Optional
from unittest.mock import MagicMock

import matplotlib
import numpy as np
import pytest
from artifact_experiment.libs.tracking.mlflow.adapter import MlflowNativeRun, MlflowRunAdapter
from artifact_experiment.libs.tracking.mlflow.client import MlflowTrackingClient
from pytest_mock import MockerFixture

matplotlib.use("Agg")


@pytest.fixture
def mock_mlflow_native_client(mocker: MockerFixture) -> MagicMock:
    mock_client = mocker.Mock()
    mock_client.get_experiment.return_value = mocker.Mock(name="test_experiment")
    mock_client.create_run.return_value = mocker.Mock()
    mock_client.set_terminated.return_value = None
    mock_client.log_artifact.return_value = None
    mock_client.log_metric.return_value = None
    mock_client.list_artifacts.return_value = []
    mock_client.get_metric_history.return_value = []
    mock_client.search_runs.return_value = []
    mock_client.get_run.return_value = mocker.Mock()
    return mock_client


@pytest.fixture
def mock_mlflow_native_run(mocker: MockerFixture) -> MagicMock:
    mock_run = mocker.Mock()
    mock_run.info.experiment_id = "test_experiment"
    mock_run.info.run_id = "test_run_uuid"
    mock_run.info.run_name = "test_run"
    mock_run.info.status = "RUNNING"
    return mock_run


@pytest.fixture
def native_run_factory(
    mock_mlflow_native_client: MagicMock, mock_mlflow_native_run: MagicMock
) -> Callable[[Optional[str], Optional[str]], MlflowNativeRun]:
    def _factory(
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> MlflowNativeRun:
        experiment_id = experiment_id or "test_experiment"
        run_id = run_id or "test_run"
        mock_mlflow_native_client.get_experiment.return_value.name = experiment_id
        mock_mlflow_native_run.info.experiment_id = experiment_id
        mock_mlflow_native_run.info.run_id = f"{run_id}_uuid"
        mock_mlflow_native_run.info.run_name = run_id
        return MlflowNativeRun(client=mock_mlflow_native_client, run=mock_mlflow_native_run)

    return _factory


@pytest.fixture
def adapter_factory(
    native_run_factory: Callable[[Optional[str], Optional[str]], MlflowNativeRun],
) -> Callable[[Optional[str], Optional[str]], MlflowRunAdapter]:
    def _factory(
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> MlflowRunAdapter:
        native_run = native_run_factory(experiment_id, run_id)
        return MlflowRunAdapter.from_native_run(native_run=native_run)

    return _factory


@pytest.fixture
def populated_adapter(
    request, adapter_factory: Callable[[Optional[str], Optional[str]], MlflowRunAdapter]
) -> MlflowRunAdapter:
    adapter = adapter_factory(None, None)
    fixture_names = request.param
    client = MlflowTrackingClient(run=adapter)
    score_idx = array_idx = plot_idx = collection_idx = 1
    for fixture_name in fixture_names:
        artifact = request.getfixturevalue(fixture_name)
        if isinstance(artifact, float):
            client.log_score(score=artifact, name=f"test_score_{score_idx}")
            score_idx += 1
        elif isinstance(artifact, np.ndarray):
            client.log_array(array=artifact, name=f"test_array_{array_idx}")
            array_idx += 1
        elif hasattr(artifact, "add_subplot"):
            client.log_plot(plot=artifact, name=f"test_plot_{plot_idx}")
            plot_idx += 1
        elif isinstance(artifact, dict):
            values = artifact.values()
            if all(isinstance(v, float) for v in values):
                client.log_score_collection(
                    score_collection=artifact, name=f"test_score_collection_{collection_idx}"
                )
                collection_idx += 1
            elif all(isinstance(v, np.ndarray) for v in values):
                client.log_array_collection(
                    array_collection=artifact, name=f"test_array_collection_{collection_idx}"
                )
                collection_idx += 1
            elif all(hasattr(v, "add_subplot") for v in values):
                client.log_plot_collection(
                    plot_collection=artifact, name=f"test_plot_collection_{collection_idx}"
                )
                collection_idx += 1
    return adapter


@pytest.fixture
def client_factory(
    adapter_factory: Callable[[Optional[str], Optional[str]], MlflowRunAdapter],
) -> Callable[[Optional[str], Optional[str]], MlflowTrackingClient]:
    def _factory(
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> MlflowTrackingClient:
        adapter = adapter_factory(experiment_id, run_id)
        return MlflowTrackingClient(run=adapter)

    return _factory
