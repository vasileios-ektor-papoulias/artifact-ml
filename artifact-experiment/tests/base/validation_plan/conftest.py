from typing import Dict
from unittest.mock import MagicMock

import pytest
from artifact_experiment.base.callbacks.artifact import ArtifactCallbackResources
from artifact_experiment.base.tracking.client import TrackingClient
from pytest_mock import MockerFixture

from tests.base.validation_plan.dummy import DummyArtifactResources, DummyResourceSpec


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
def resource_spec() -> DummyResourceSpec:
    return DummyResourceSpec()


@pytest.fixture
def artifact_resources() -> DummyArtifactResources:
    return DummyArtifactResources()


@pytest.fixture
def callback_resources(artifact_resources: DummyArtifactResources) -> ArtifactCallbackResources:
    return ArtifactCallbackResources(artifact_resources=artifact_resources)


@pytest.fixture
def mock_handlers(mocker: MockerFixture) -> Dict[str, MagicMock]:
    handlers = {}
    for handler_type in [
        "score",
        "array",
        "plot",
        "score_collection",
        "array_collection",
        "plot_collection",
    ]:
        handler = mocker.Mock()
        handler.active_cache = {}
        handler.execute = mocker.Mock()
        handler.clear = mocker.Mock()
        handlers[handler_type] = handler
    return handlers
