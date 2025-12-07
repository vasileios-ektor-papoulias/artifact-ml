from typing import Any, Type, Union

import pytest
from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._base.orchestration.repository import (
    ArtifactHyperparamsRepository,
    ArtifactRepository,
)
from artifact_core._base.orchestration.repository_writer import ArtifactRepositoryWriter
from artifact_core._base.typing.artifact_result import Score

from tests._base.dummy.artifacts.base import DummyArtifact
from tests._base.dummy.artifacts.scores.unregistered import (
    UnregisteredArtifact,
    UnregisteredArtifactHyperparams,
)
from tests._base.dummy.resource_spec import DummyResourceSpec
from tests._base.dummy.resources import DummyArtifactResources
from tests._base.dummy.types.scores import DummyScoreType


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, artifact_class, expected_key",
    [
        (
            DummyScoreType.NOT_REGISTERED,
            UnregisteredArtifact,
            "NOT_REGISTERED",
        ),
        (
            DummyScoreType.IN_ALTERNATIVE_REGISTRY,
            UnregisteredArtifact,
            "IN_ALTERNATIVE_REGISTRY",
        ),
        ("NEW_ARTIFACT", UnregisteredArtifact, "NEW_ARTIFACT"),
        ("ANOTHER_NEW_ARTIFACT", UnregisteredArtifact, "ANOTHER_NEW_ARTIFACT"),
    ],
)
def test_register_artifact(
    artifact_type: Union[DummyScoreType, str],
    artifact_class: Type[DummyArtifact[Any, Any]],
    expected_key: str,
):
    repository: ArtifactRepository[DummyArtifactResources, DummyResourceSpec, Score] = {}
    decorator = ArtifactRepositoryWriter.put_artifact(
        artifact_type=artifact_type, repository=repository
    )
    result = decorator(artifact_class)
    assert result is artifact_class
    assert expected_key in repository
    assert repository[expected_key] is artifact_class


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, artifact_class, expected_key, expected_warning_message",
    [
        (
            DummyScoreType.DUMMY_SCORE_ARTIFACT,
            UnregisteredArtifact,
            "DUMMY_SCORE_ARTIFACT",
            "Artifact already registered for artifact_type=" + "DUMMY_SCORE_ARTIFACT",
        ),
        (
            "EXISTING_ARTIFACT",
            UnregisteredArtifact,
            "EXISTING_ARTIFACT",
            "Artifact already registered for artifact_type=" + "EXISTING_ARTIFACT",
        ),
    ],
)
def test_register_artifact_already_registered(
    artifact_type: Union[DummyScoreType, str],
    artifact_class: Type[DummyArtifact[Any, Any]],
    expected_key: str,
    expected_warning_message: str,
):
    existing_class = UnregisteredArtifact
    repository: ArtifactRepository[DummyArtifactResources, DummyResourceSpec, Score] = {
        expected_key: existing_class
    }
    with pytest.warns(UserWarning, match=expected_warning_message):
        decorator = ArtifactRepositoryWriter.put_artifact(
            artifact_type=artifact_type, repository=repository
        )
        result = decorator(artifact_class)
    assert result is artifact_class
    assert repository[expected_key] is existing_class


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_types, expected_keys",
    [
        (["ARTIFACT_1", "ARTIFACT_2"], ["ARTIFACT_1", "ARTIFACT_2"]),
        (
            ["ARTIFACT_1", "ARTIFACT_2", "ARTIFACT_3"],
            ["ARTIFACT_1", "ARTIFACT_2", "ARTIFACT_3"],
        ),
        (
            [DummyScoreType.NOT_REGISTERED, DummyScoreType.IN_ALTERNATIVE_REGISTRY],
            ["NOT_REGISTERED", "IN_ALTERNATIVE_REGISTRY"],
        ),
        (
            [DummyScoreType.NOT_REGISTERED, "CUSTOM_ARTIFACT", "ANOTHER_ARTIFACT"],
            ["NOT_REGISTERED", "CUSTOM_ARTIFACT", "ANOTHER_ARTIFACT"],
        ),
        (
            ["A", "B", "C", "D", "E"],
            ["A", "B", "C", "D", "E"],
        ),
    ],
)
def test_register_artifact_multiple(
    artifact_types: list[Union[DummyScoreType, str]], expected_keys: list[str]
):
    repository: ArtifactRepository[DummyArtifactResources, DummyResourceSpec, Score] = {}

    for artifact_type in artifact_types:
        decorator = ArtifactRepositoryWriter.put_artifact(
            artifact_type=artifact_type, repository=repository
        )
        decorator(UnregisteredArtifact)

    assert len(repository) == len(expected_keys)
    for key in expected_keys:
        assert key in repository
        assert repository[key] is UnregisteredArtifact


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, hyperparams_class, expected_key",
    [
        (
            DummyScoreType.NOT_REGISTERED,
            UnregisteredArtifactHyperparams,
            "NOT_REGISTERED",
        ),
        (
            DummyScoreType.IN_ALTERNATIVE_REGISTRY,
            UnregisteredArtifactHyperparams,
            "IN_ALTERNATIVE_REGISTRY",
        ),
        ("NEW_HYPERPARAMS", UnregisteredArtifactHyperparams, "NEW_HYPERPARAMS"),
        ("ANOTHER_NEW_HYPERPARAMS", UnregisteredArtifactHyperparams, "ANOTHER_NEW_HYPERPARAMS"),
    ],
)
def test_register_artifact_hyperparams(
    artifact_type: Union[DummyScoreType, str],
    hyperparams_class: Type[ArtifactHyperparams],
    expected_key: str,
):
    repository: ArtifactHyperparamsRepository = {}
    decorator = ArtifactRepositoryWriter.put_artifact_hyperparams(
        artifact_type=artifact_type, repository=repository
    )
    result = decorator(hyperparams_class)
    assert result is hyperparams_class
    assert expected_key in repository
    assert repository[expected_key] is hyperparams_class


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, hyperparams_class, expected_key, expected_warning_message",
    [
        (
            DummyScoreType.DUMMY_SCORE_ARTIFACT,
            UnregisteredArtifactHyperparams,
            "DUMMY_SCORE_ARTIFACT",
            "Hyperparams already registered for artifact_type=DUMMY_SCORE_ARTIFACT",
        ),
        (
            "EXISTING_HYPERPARAMS",
            UnregisteredArtifactHyperparams,
            "EXISTING_HYPERPARAMS",
            "Hyperparams already registered for artifact_type=EXISTING_HYPERPARAMS",
        ),
    ],
)
def test_register_artifact_hyperparams_already_registered(
    artifact_type: Union[DummyScoreType, str],
    hyperparams_class: Type[ArtifactHyperparams],
    expected_key: str,
    expected_warning_message: str,
):
    existing_class = UnregisteredArtifactHyperparams
    repository: ArtifactHyperparamsRepository = {expected_key: existing_class}
    with pytest.warns(UserWarning, match=expected_warning_message):
        decorator = ArtifactRepositoryWriter.put_artifact_hyperparams(
            artifact_type=artifact_type, repository=repository
        )
        result = decorator(hyperparams_class)
    assert result is hyperparams_class
    assert repository[expected_key] is existing_class


@pytest.mark.unit
@pytest.mark.parametrize(
    "hyperparams_types, expected_keys",
    [
        (["HYPERPARAMS_1", "HYPERPARAMS_2"], ["HYPERPARAMS_1", "HYPERPARAMS_2"]),
        (
            ["HYPERPARAMS_1", "HYPERPARAMS_2", "HYPERPARAMS_3"],
            ["HYPERPARAMS_1", "HYPERPARAMS_2", "HYPERPARAMS_3"],
        ),
        (
            [DummyScoreType.NOT_REGISTERED, DummyScoreType.IN_ALTERNATIVE_REGISTRY],
            ["NOT_REGISTERED", "IN_ALTERNATIVE_REGISTRY"],
        ),
        (
            [DummyScoreType.NOT_REGISTERED, "CUSTOM_HYPERPARAMS"],
            ["NOT_REGISTERED", "CUSTOM_HYPERPARAMS"],
        ),
    ],
)
def test_register_hyperparams_multiple(
    hyperparams_types: list[Union[DummyScoreType, str]], expected_keys: list[str]
):
    repository: ArtifactHyperparamsRepository = {}

    for hyperparams_type in hyperparams_types:
        decorator = ArtifactRepositoryWriter.put_artifact_hyperparams(
            artifact_type=hyperparams_type, repository=repository
        )
        decorator(UnregisteredArtifactHyperparams)

    assert len(repository) == len(expected_keys)
    for key in expected_keys:
        assert key in repository
        assert repository[key] is UnregisteredArtifactHyperparams
