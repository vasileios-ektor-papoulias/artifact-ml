from typing import List
from unittest.mock import MagicMock

import pytest
from artifact_experiment.base.callbacks.base import CallbackResources
from artifact_experiment.base.tracking.client import TrackingClient
from pytest_mock import MockerFixture


@pytest.fixture
def mock_tracking_client(mocker: MockerFixture) -> MagicMock:
    client = mocker.Mock(spec=TrackingClient)
    client.log_score = mocker.Mock()
    client.log_array = mocker.Mock()
    client.log_plot = mocker.Mock()
    client.log_score_collection = mocker.Mock()
    client.log_array_collection = mocker.Mock()
    client.log_plot_collection = mocker.Mock()
    return client


@pytest.fixture
def test_resources() -> CallbackResources:
    return CallbackResources()


@pytest.fixture
def callback_key() -> str:
    return "test_callback"


@pytest.fixture
def multiple_callback_keys() -> List[str]:
    return ["callback1", "callback2", "callback3"]
