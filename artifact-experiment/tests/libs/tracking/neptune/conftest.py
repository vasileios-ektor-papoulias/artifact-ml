from typing import Callable, Optional
from unittest.mock import MagicMock

import matplotlib
import numpy as np
import pytest
from artifact_experiment.libs.tracking.neptune.adapter import NeptuneRunAdapter
from artifact_experiment.libs.tracking.neptune.client import NeptuneTrackingClient
from pytest_mock import MockerFixture

matplotlib.use("Agg")


@pytest.fixture
def native_run_factory(
    mocker: MockerFixture,
) -> Callable[[Optional[str], Optional[str]], MagicMock]:
    def _factory(experiment_id: Optional[str] = None, run_id: Optional[str] = None) -> MagicMock:
        experiment_id = experiment_id or "test_experiment"
        run_id = run_id or "test_run"
        mock_run = mocker.Mock()
        mock_run.__getitem__ = mocker.Mock()
        mock_run.__setitem__ = mocker.Mock()
        mock_sys_experiment_name = mocker.Mock()
        mock_sys_experiment_name.fetch.return_value = experiment_id
        mock_sys_id = mocker.Mock()
        mock_sys_id.fetch.return_value = run_id
        mock_run.__getitem__.side_effect = lambda key: {
            "sys/experiment/name": mock_sys_experiment_name,
            "sys/id": mock_sys_id,
        }.get(key, mocker.Mock())
        mock_run.stop.return_value = None
        mock_run.fetch.return_value = {"sys": {"state": "running"}}
        return mock_run

    return _factory


@pytest.fixture
def adapter_factory(
    native_run_factory: Callable[[Optional[str], Optional[str]], MagicMock],
) -> Callable[[Optional[str], Optional[str]], NeptuneRunAdapter]:
    def _factory(
        experiment_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> NeptuneRunAdapter:
        native_run = native_run_factory(experiment_id, run_id)
        return NeptuneRunAdapter.from_native_run(native_run=native_run)

    return _factory


@pytest.fixture
def populated_adapter(
    request, adapter_factory: Callable[[Optional[str], Optional[str]], NeptuneRunAdapter]
) -> NeptuneRunAdapter:
    adapter = adapter_factory(None, None)
    fixture_names = request.param
    client = NeptuneTrackingClient(run=adapter)
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
    adapter_factory: Callable[[Optional[str], Optional[str]], NeptuneRunAdapter],
) -> Callable[[Optional[str], Optional[str]], NeptuneTrackingClient]:
    def _factory(
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> NeptuneTrackingClient:
        adapter = adapter_factory(experiment_id, run_id)
        return NeptuneTrackingClient(run=adapter)

    return _factory
