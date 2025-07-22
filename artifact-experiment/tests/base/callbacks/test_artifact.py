from typing import Callable, Optional, Tuple
from unittest.mock import MagicMock

import pytest
from artifact_core.base.artifact_dependencies import ArtifactResult
from artifact_experiment.base.callbacks.artifact import (
    ArtifactArrayCallback,
    ArtifactArrayCollectionCallback,
    ArtifactCallback,
    ArtifactCallbackResources,
    ArtifactPlotCallback,
    ArtifactPlotCollectionCallback,
    ArtifactScoreCallback,
    ArtifactScoreCollectionCallback,
)
from artifact_experiment.base.tracking.client import TrackingClient
from pytest_mock import MockerFixture


@pytest.fixture
def resources_factory(
    mocker: MockerFixture,
) -> Callable[[], Tuple[ArtifactCallbackResources, MagicMock]]:
    def _factory() -> Tuple[ArtifactCallbackResources, MagicMock]:
        artifact_resources = mocker.Mock()
        callback_resources = ArtifactCallbackResources(artifact_resources=artifact_resources)
        return callback_resources, artifact_resources

    return _factory


@pytest.fixture
def artifact_factory(mocker: MockerFixture) -> Callable[[ArtifactResult], MagicMock]:
    def _factory(compute_return_value: ArtifactResult) -> MagicMock:
        artifact = mocker.Mock()
        artifact.compute.return_value = compute_return_value
        return artifact

    return _factory


@pytest.fixture
def callback_factory(
    artifact_factory: Callable[[ArtifactResult], MagicMock],
) -> Callable[
    [str, str, ArtifactResult, Optional[TrackingClient]], Tuple[ArtifactCallback, MagicMock]
]:
    def _factory(
        callback_type: str,
        callback_key: str,
        return_value: ArtifactResult,
        tracking_client: Optional[TrackingClient],
    ) -> Tuple[ArtifactCallback, MagicMock]:
        artifact = artifact_factory(return_value)
        if callback_type == "score":
            callback = ArtifactScoreCallback(
                key=callback_key, artifact=artifact, tracking_client=tracking_client
            )
        elif callback_type == "array":
            callback = ArtifactArrayCallback(
                key=callback_key, artifact=artifact, tracking_client=tracking_client
            )
        elif callback_type == "plot":
            callback = ArtifactPlotCallback(
                key=callback_key, artifact=artifact, tracking_client=tracking_client
            )
        elif callback_type == "score_collection":
            callback = ArtifactScoreCollectionCallback(
                key=callback_key, artifact=artifact, tracking_client=tracking_client
            )
        elif callback_type == "array_collection":
            callback = ArtifactArrayCollectionCallback(
                key=callback_key, artifact=artifact, tracking_client=tracking_client
            )
        elif callback_type == "plot_collection":
            callback = ArtifactPlotCollectionCallback(
                key=callback_key, artifact=artifact, tracking_client=tracking_client
            )
        else:
            raise ValueError(f"Unknown callback type: {callback_type}")
        return callback, artifact

    return _factory


@pytest.fixture
def return_value(request) -> ArtifactResult:
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    "callback_type, return_value, has_tracking_client",
    [
        ("score", "score_1", True),
        ("score", "score_1", False),
        ("score", "score_2", True),
        ("score", "score_2", False),
        ("array", "array_1", True),
        ("array", "array_1", False),
        ("array", "array_2", True),
        ("array", "array_2", False),
        ("plot", "plot_1", True),
        ("plot", "plot_1", False),
        ("plot", "plot_2", True),
        ("plot", "plot_2", False),
        ("score_collection", "score_collection_1", True),
        ("score_collection", "score_collection_1", False),
        ("score_collection", "score_collection_2", True),
        ("score_collection", "score_collection_2", False),
        ("array_collection", "array_collection_1", True),
        ("array_collection", "array_collection_1", False),
        ("array_collection", "array_collection_2", True),
        ("array_collection", "array_collection_2", False),
        ("plot_collection", "plot_collection_1", True),
        ("plot_collection", "plot_collection_1", False),
        ("plot_collection", "plot_collection_2", True),
        ("plot_collection", "plot_collection_2", False),
    ],
    indirect=["return_value"],
)
def test_execute(
    resources_factory: Callable[[], Tuple[ArtifactCallbackResources, MagicMock]],
    callback_factory: Callable[
        [str, str, ArtifactResult, Optional[TrackingClient]], Tuple[ArtifactCallback, MagicMock]
    ],
    mock_tracking_client_factory: Callable[[], MagicMock],
    callback_type: str,
    return_value: ArtifactResult,
    has_tracking_client: bool,
):
    callback_key = "test_key"
    callback_resources, artifact_resources = resources_factory()
    assert callback_resources.artifact_resources == artifact_resources
    tracking_client = mock_tracking_client_factory() if has_tracking_client else None
    callback, artifact = callback_factory(
        callback_type, callback_key, return_value, tracking_client
    )
    callback.execute(resources=callback_resources)
    artifact.compute.assert_called_once_with(resources=artifact_resources)
    assert callback.value is return_value
    if tracking_client is not None:
        log_method = getattr(tracking_client, "log_" + callback_type)
        log_method.assert_called_once()
        log_method_args, log_method_kwargs = log_method.call_args
        assert log_method_args == ()
        assert list(log_method_kwargs.values()) == [callback.value, callback.key]
