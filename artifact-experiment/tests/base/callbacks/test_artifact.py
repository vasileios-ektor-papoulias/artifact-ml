from typing import Callable, List, Tuple, Type
from unittest.mock import MagicMock

import numpy as np
import pytest
from artifact_core.base.artifact_dependencies import ArtifactResult
from artifact_experiment.base.callbacks.artifact import (
    ArtifactArrayCallback,
    ArtifactCallback,
    ArtifactCallbackResources,
    ArtifactScoreCallback,
    ArtifactScoreCollectionCallback,
)
from artifact_experiment.base.callbacks.base import CallbackResources
from pytest_mock import MockerFixture


@pytest.fixture
def artifact_factory(mocker: MockerFixture) -> Callable[[ArtifactResult], MagicMock]:
    def _factory(compute_return_value: ArtifactResult) -> MagicMock:
        artifact = mocker.Mock()
        artifact.compute.return_value = compute_return_value
        return artifact

    return _factory


@pytest.fixture
def resources_factory(
    mocker: MockerFixture,
) -> Callable[[], Tuple[ArtifactCallbackResources, MagicMock]]:
    def _factory() -> Tuple[ArtifactCallbackResources, MagicMock]:
        artifact_resources = mocker.Mock()
        callback_resources = ArtifactCallbackResources(artifact_resources=artifact_resources)
        return callback_resources, artifact_resources

    return _factory


def test_resources(
    mocker: MockerFixture,
):
    artifact_resources = mocker.Mock()
    artifact_resources.some_property = "test_value"
    callback_resources = ArtifactCallbackResources(artifact_resources=artifact_resources)
    assert isinstance(callback_resources, CallbackResources)
    assert callback_resources.artifact_resources == artifact_resources
    assert callback_resources.artifact_resources.some_property == "test_value"


@pytest.mark.parametrize(
    "callback_class, expected_compute_value, expected_log_method",
    [
        (ArtifactScoreCallback, 42.5, "log_score"),
        (ArtifactArrayCallback, np.array([1, 2, 3]), "log_array"),
        (ArtifactScoreCollectionCallback, {"metric1": 1.0, "metric2": 2.0}, "log_score_collection"),
    ],
)
def test_execute(
    mock_tracking_client: MagicMock,
    callback_key: str,
    artifact_factory: Callable[[ArtifactResult], MagicMock],
    resources_factory: Callable[[], Tuple[ArtifactCallbackResources, MagicMock]],
    callback_class: Type[ArtifactCallback],
    expected_compute_value: ArtifactResult,
    expected_log_method: str,
):
    artifact = artifact_factory(expected_compute_value)
    tracking_client = mock_tracking_client
    callback_resources, artifact_resources = resources_factory()
    callback = callback_class(key=callback_key, artifact=artifact, tracking_client=tracking_client)
    callback.execute(resources=callback_resources)
    artifact.compute.assert_called_once_with(resources=artifact_resources)
    assert callback.value is not None
    if isinstance(expected_compute_value, np.ndarray):
        assert np.array_equal(callback.value, expected_compute_value)
    else:
        assert callback.value == expected_compute_value
    log_method = getattr(tracking_client, expected_log_method)
    if expected_log_method == "log_score":
        log_method.assert_called_once_with(score=expected_compute_value, name=callback_key)
    elif expected_log_method == "log_array":
        log_method.assert_called_once_with(array=expected_compute_value, name=callback_key)
    elif expected_log_method == "log_score_collection":
        log_method.assert_called_once_with(
            score_collection=expected_compute_value, name=callback_key
        )


@pytest.mark.parametrize(
    "callback_class, compute_value",
    [
        (ArtifactScoreCallback, 5.0),
        (ArtifactArrayCallback, np.array([10, 20])),
        (ArtifactScoreCollectionCallback, {"test": 99.9}),
    ],
)
def test_no_tracking_client(
    callback_key: str,
    artifact_factory: Callable[[ArtifactResult], MagicMock],
    resources_factory: Callable[[], Tuple[ArtifactCallbackResources, MagicMock]],
    callback_class: Type[ArtifactCallback],
    compute_value: ArtifactResult,
):
    artifact = artifact_factory(compute_value)
    callback_resources, artifact_resources = resources_factory()
    callback = callback_class(key=callback_key, artifact=artifact, tracking_client=None)
    callback.execute(resources=callback_resources)
    artifact.compute.assert_called_once_with(resources=artifact_resources)
    assert callback.value is not None
    if isinstance(compute_value, np.ndarray):
        assert np.array_equal(callback.value, compute_value)
    else:
        assert callback.value == compute_value


@pytest.mark.parametrize(
    "compute_values",
    [
        [1.0, 2.0, 3.0],
        [0.0, -1.0, 99.99],
        [42.0],
    ],
)
def test_multiple_executions(
    callback_key: str,
    artifact_factory: Callable[[ArtifactResult], MagicMock],
    resources_factory: Callable[[], Tuple[ArtifactCallbackResources, MagicMock]],
    compute_values: List[ArtifactResult],
):
    callback_resources, artifact_resources = resources_factory()
    for compute_value in compute_values:
        artifact = artifact_factory(compute_value)
        callback = ArtifactScoreCallback(key=callback_key, artifact=artifact, tracking_client=None)
        callback.execute(resources=callback_resources)
        assert callback.value == compute_value
        artifact.compute.assert_called_once_with(resources=artifact_resources)
