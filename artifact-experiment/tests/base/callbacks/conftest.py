from typing import Callable
from unittest.mock import MagicMock

import pytest
from artifact_experiment.base.tracking.backend.client import TrackingClient
from pytest_mock import MockerFixture


@pytest.fixture
def mock_tracking_client_factory(mocker: MockerFixture) -> Callable[[], MagicMock]:
    def _factory() -> MagicMock:
        client = mocker.Mock(spec=TrackingClient)
        client.log_score = mocker.Mock()
        client.log_array = mocker.Mock()
        client.log_plot = mocker.Mock()
        client.log_score_collection = mocker.Mock()
        client.log_array_collection = mocker.Mock()
        client.log_plot_collection = mocker.Mock()
        return client

    return _factory
