from typing import Callable, Optional

import matplotlib
import numpy as np
import pytest
from artifact_experiment.libs.tracking.filesystem.adapter import FilesystemRunAdapter
from artifact_experiment.libs.tracking.filesystem.client import FilesystemTrackingClient
from artifact_experiment.libs.tracking.filesystem.native_run import FilesystemRun
from pytest_mock import MockerFixture

matplotlib.use("Agg")


@pytest.fixture
def native_run_factory(
    mocker: MockerFixture,
) -> Callable[[Optional[str], Optional[str]], FilesystemRun]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> FilesystemRun:
        experiment_id = experiment_id or "test_experiment"
        run_id = run_id or "test_run"
        mocker.patch("os.makedirs")
        mocker.patch(
            "artifact_experiment.libs.utils.directory_opener.DirectoryOpenButton", new=None
        )
        mock_run = mocker.create_autospec(FilesystemRun, instance=True)
        mock_run.experiment_id = experiment_id
        mock_run.id = run_id
        mock_run.is_active = True
        mock_run.experiment_dir = f"/mock/path/{experiment_id}"
        mock_run.run_dir = f"/mock/path/{experiment_id}/{run_id}"
        mock_run.stop.return_value = None
        return mock_run

    return _factory


@pytest.fixture
def adapter_factory(
    native_run_factory: Callable[[Optional[str], Optional[str]], FilesystemRun],
) -> Callable[[Optional[str], Optional[str]], FilesystemRunAdapter]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> FilesystemRunAdapter:
        experiment_id = experiment_id or "test_experiment"
        run_id = run_id or "test_run"
        native_run = native_run_factory(experiment_id, run_id)
        return FilesystemRunAdapter.from_native_run(native_run=native_run)

    return _factory


@pytest.fixture
def populated_adapter(
    request, adapter_factory: Callable[[Optional[str], Optional[str]], FilesystemRunAdapter]
) -> FilesystemRunAdapter:
    adapter = adapter_factory(None, None)
    fixture_names = request.param
    client = FilesystemTrackingClient(run=adapter)
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
    adapter_factory: Callable[[Optional[str], Optional[str]], FilesystemRunAdapter],
) -> Callable[[Optional[str], Optional[str]], FilesystemTrackingClient]:
    def _factory(
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> FilesystemTrackingClient:
        adapter = adapter_factory(experiment_id, run_id)
        return FilesystemTrackingClient(run=adapter)

    return _factory
