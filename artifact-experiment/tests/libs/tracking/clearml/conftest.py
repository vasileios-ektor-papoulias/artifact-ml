from typing import Callable, Optional

import matplotlib
import numpy as np
import pytest
from artifact_experiment.libs.tracking.clear_ml.adapter import ClearMLRunAdapter
from artifact_experiment.libs.tracking.clear_ml.client import ClearMLTrackingClient
from clearml import Task
from pytest_mock import MockerFixture

matplotlib.use("Agg")


@pytest.fixture
def native_run_factory(mocker: MockerFixture) -> Callable[[Optional[str], Optional[str]], Task]:
    def _factory(experiment_id: Optional[str] = None, run_id: Optional[str] = None) -> Task:
        experiment_id = experiment_id or "test_experiment"
        run_id = run_id or "test_run"
        mock_task = mocker.Mock()
        mock_task.project = experiment_id
        mock_task.id = run_id
        mock_task.get_status.return_value = "queued"
        mock_task.close.return_value = None
        mock_task.upload_artifact.return_value = None
        mock_task.artifacts = {}
        mock_task.get_reported_scalars.return_value = {}
        mock_task.get_reported_plots.return_value = {}
        mock_logger = mocker.Mock()
        mock_logger.report_scalar.return_value = None
        mock_logger.report_matplotlib_figure.return_value = None
        mock_task.get_logger.return_value = mock_logger
        return mock_task

    return _factory


@pytest.fixture
def adapter_factory(
    native_run_factory: Callable[[Optional[str], Optional[str]], Task],
) -> Callable[[Optional[str], Optional[str]], ClearMLRunAdapter]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> ClearMLRunAdapter:
        experiment_id = experiment_id or "test_experiment"
        run_id = run_id or "test_run"
        native_run = native_run_factory(experiment_id, run_id)
        return ClearMLRunAdapter.from_native_run(native_run=native_run)

    return _factory


@pytest.fixture
def populated_adapter(
    request, adapter_factory: Callable[[Optional[str], Optional[str]], ClearMLRunAdapter]
) -> ClearMLRunAdapter:
    adapter = adapter_factory(None, None)
    fixture_names = request.param

    client = ClearMLTrackingClient(run=adapter)

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
    adapter_factory: Callable[[Optional[str], Optional[str]], ClearMLRunAdapter],
) -> Callable[[Optional[str], Optional[str]], ClearMLTrackingClient]:
    def _factory(
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> ClearMLTrackingClient:
        adapter = adapter_factory(experiment_id, run_id)
        return ClearMLTrackingClient(run=adapter)

    return _factory
