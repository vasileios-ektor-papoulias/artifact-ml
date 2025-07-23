from typing import Callable, Dict, Tuple
from unittest.mock import MagicMock

import pytest
from artifact_experiment.base.callbacks.artifact import ArtifactCallbackResources
from artifact_experiment.base.tracking.client import TrackingClient
from pytest_mock import MockerFixture

from tests.base.validation_plan.dummy.validation_plan import (
    DummyArtifactResources,
    DummyResourceSpec,
)


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
def mock_callback_handlers(mocker: MockerFixture) -> Dict[str, MagicMock]:
    dict_handlers = {}
    for handler_type in [
        "scores",
        "arrays",
        "plots",
        "score_collections",
        "array_collections",
        "plot_collections",
    ]:
        handler = mocker.Mock()
        handler.tracking_client = None
        handler.active_cache = {}
        handler.execute = mocker.Mock()
        handler.clear = mocker.Mock()
        dict_handlers[handler_type] = handler
    return dict_handlers


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


@pytest.fixture
def resources_factory() -> Callable[
    [], Tuple[ArtifactCallbackResources, DummyArtifactResources, DummyResourceSpec]
]:
    def _factory() -> Tuple[ArtifactCallbackResources, DummyArtifactResources, DummyResourceSpec]:
        artifact_resources = DummyArtifactResources()
        callback_resources = ArtifactCallbackResources(artifact_resources=artifact_resources)
        resource_spec = DummyResourceSpec()
        return callback_resources, artifact_resources, resource_spec

    return _factory
