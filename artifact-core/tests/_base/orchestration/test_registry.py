from typing import Any, Type, Union

import pytest
from artifact_core._base.core.hyperparams import (
    NO_ARTIFACT_HYPERPARAMS,
    ArtifactHyperparams,
    NoArtifactHyperparams,
)
from artifact_core._base.typing.artifact_result import ArtifactResult, Score

from tests._base.dummy.artifacts.base import DummyArtifact
from tests._base.dummy.artifacts.scores.alternative import AlternativeRegistryArtifact
from tests._base.dummy.artifacts.scores.custom import CustomScoreArtifact, CustomScoreHyperparams
from tests._base.dummy.artifacts.scores.dummy import DummyScoreArtifact, DummyScoreHyperparams
from tests._base.dummy.artifacts.scores.no_hyperparams import NoHyperparamsArtifact
from tests._base.dummy.artifacts.scores.no_hyperparams_custom import (
    NoHyperparamsCustomScoreArtifact,
)
from tests._base.dummy.artifacts.scores.unregistered import (
    UnregisteredArtifact,
    UnregisteredArtifactHyperparams,
)
from tests._base.dummy.registries.base import DummyArtifactRegistry
from tests._base.dummy.registries.scores import (
    AlternativeDummyScoreRegistry,
    DummyScoreRegistry,
    InvalidParamDummyScoreRegistry,
    MissingParamDummyScoreRegistry,
)
from tests._base.dummy.resource_spec import DummyResourceSpec
from tests._base.dummy.types.scores import DummyScoreType


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_registry, artifact_type, resource_spec, expected_artifact_class, expected_hyperparams",
    [
        (
            DummyScoreRegistry,
            DummyScoreType.DUMMY_SCORE_ARTIFACT,
            DummyResourceSpec(scale=1),
            DummyScoreArtifact,
            DummyScoreHyperparams(adjust_scale=True),
        ),
        (
            DummyScoreRegistry,
            DummyScoreType.DUMMY_SCORE_ARTIFACT,
            DummyResourceSpec(scale=10),
            DummyScoreArtifact,
            DummyScoreHyperparams(adjust_scale=True),
        ),
        (
            DummyScoreRegistry,
            DummyScoreType.NO_HYPERPARAMS_ARTIFACT,
            DummyResourceSpec(scale=1),
            NoHyperparamsArtifact,
            NO_ARTIFACT_HYPERPARAMS,
        ),
        (
            AlternativeDummyScoreRegistry,
            DummyScoreType.IN_ALTERNATIVE_REGISTRY,
            DummyResourceSpec(scale=1),
            AlternativeRegistryArtifact,
            NO_ARTIFACT_HYPERPARAMS,
        ),
        (
            DummyScoreRegistry,
            "CUSTOM_SCORE_ARTIFACT",
            DummyResourceSpec(scale=1),
            CustomScoreArtifact,
            CustomScoreHyperparams(result=0),
        ),
        (
            DummyScoreRegistry,
            "NO_HYPERPARAMS_CUSTOM_SCORE_ARTIFACT",
            DummyResourceSpec(scale=1),
            NoHyperparamsCustomScoreArtifact,
            NO_ARTIFACT_HYPERPARAMS,
        ),
    ],
)
def test_get(
    artifact_registry: Type[DummyArtifactRegistry[DummyScoreType, Score]],
    artifact_type: Union[DummyScoreType, str],
    resource_spec: DummyResourceSpec,
    expected_artifact_class: Type[DummyArtifact[Any, ArtifactResult]],
    expected_hyperparams: Union[DummyScoreHyperparams, NoArtifactHyperparams],
):
    artifact = artifact_registry.get(artifact_type=artifact_type, resource_spec=resource_spec)
    assert isinstance(artifact, expected_artifact_class)
    assert artifact.resource_spec == resource_spec
    assert artifact.hyperparams == expected_hyperparams


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_registry, artifact_type, resource_spec",
    [
        (
            DummyScoreRegistry,
            DummyScoreType.IN_ALTERNATIVE_REGISTRY,
            DummyResourceSpec(scale=1),
        ),
        (
            DummyScoreRegistry,
            DummyScoreType.NOT_REGISTERED,
            DummyResourceSpec(scale=1),
        ),
    ],
)
def test_get_unregistered_artifact(
    artifact_registry: Type[DummyArtifactRegistry[DummyScoreType, Score]],
    artifact_type: Union[DummyScoreType, str],
    resource_spec: DummyResourceSpec,
):
    str_artifact_type = artifact_type if isinstance(artifact_type, str) else artifact_type.name
    with pytest.raises(ValueError, match=f"Artifact {str_artifact_type} not registered"):
        artifact_registry.get(artifact_type=artifact_type, resource_spec=resource_spec)


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_registry, artifact_type, resource_spec, expected_hyperparams_class",
    [
        (
            AlternativeDummyScoreRegistry,
            DummyScoreType.DUMMY_SCORE_ARTIFACT,
            DummyResourceSpec(scale=1),
            DummyScoreHyperparams,
        ),
    ],
)
def test_get_missing_config(
    artifact_registry: Type[DummyArtifactRegistry[DummyScoreType, Score]],
    artifact_type: Union[DummyScoreType, str],
    resource_spec: DummyResourceSpec,
    expected_hyperparams_class: Type[ArtifactHyperparams],
):
    with pytest.raises(
        ValueError,
        match=f"Missing config for hyperparams type {expected_hyperparams_class.__name__}",
    ):
        artifact_registry.get(artifact_type=artifact_type, resource_spec=resource_spec)


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_registry, artifact_type, resource_spec, expected_hyperparams_class",
    [
        (
            MissingParamDummyScoreRegistry,
            DummyScoreType.DUMMY_SCORE_ARTIFACT,
            DummyResourceSpec(scale=1),
            DummyScoreHyperparams,
        ),
        (
            InvalidParamDummyScoreRegistry,
            DummyScoreType.DUMMY_SCORE_ARTIFACT,
            DummyResourceSpec(scale=1),
            DummyScoreHyperparams,
        ),
    ],
)
def test_get_invalid_param(
    artifact_registry: Type[DummyArtifactRegistry[DummyScoreType, Score]],
    artifact_type: Union[DummyScoreType, str],
    resource_spec: DummyResourceSpec,
    expected_hyperparams_class: Type[ArtifactHyperparams],
):
    with pytest.raises(
        ValueError,
        match=f"Error instantiating '{expected_hyperparams_class.__name__}'",
    ):
        artifact_registry.get(artifact_type=artifact_type, resource_spec=resource_spec)


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_registry, artifact_type, artifact_class, resource_spec",
    [
        (
            AlternativeDummyScoreRegistry,
            DummyScoreType.NOT_REGISTERED,
            NoHyperparamsArtifact,
            DummyResourceSpec(scale=1.0),
        ),
        (
            AlternativeDummyScoreRegistry,
            DummyScoreType.NEW_SCORE,
            UnregisteredArtifact,
            DummyResourceSpec(scale=5.0),
        ),
    ],
)
def test_register_artifact(
    artifact_registry: Type[DummyArtifactRegistry[DummyScoreType, Score]],
    artifact_type: DummyScoreType,
    artifact_class: Type[DummyArtifact[Any, ArtifactResult]],
    resource_spec: DummyResourceSpec,
):
    decorator = artifact_registry.register_artifact(artifact_type=artifact_type)
    result = decorator(artifact_class)
    assert result is artifact_class

    artifact = artifact_registry.get(artifact_type=artifact_type, resource_spec=resource_spec)
    assert isinstance(artifact, artifact_class)
    assert artifact.resource_spec == resource_spec


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_registry, artifact_type",
    [
        (DummyScoreRegistry, DummyScoreType.DUMMY_SCORE_ARTIFACT),
        (DummyScoreRegistry, DummyScoreType.NO_HYPERPARAMS_ARTIFACT),
    ],
)
def test_register_artifact_duplicate(
    artifact_registry: Type[DummyArtifactRegistry[DummyScoreType, Score]],
    artifact_type: DummyScoreType,
):
    str_artifact_type = artifact_type.name
    with pytest.warns(
        UserWarning,
        match=f"Artifact already registered for artifact_type={str_artifact_type}",
    ):
        decorator = artifact_registry.register_artifact(artifact_type)
        result = decorator(UnregisteredArtifact)
        assert result is UnregisteredArtifact


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_registry, artifact_type, hyperparams_class",
    [
        (
            AlternativeDummyScoreRegistry,
            DummyScoreType.NOT_REGISTERED,
            UnregisteredArtifactHyperparams,
        ),
        (
            AlternativeDummyScoreRegistry,
            DummyScoreType.NEW_SCORE,
            DummyScoreHyperparams,
        ),
    ],
)
def test_register_artifact_hyperparams(
    artifact_registry: Type[DummyArtifactRegistry[DummyScoreType, Score]],
    artifact_type: DummyScoreType,
    hyperparams_class: Type[ArtifactHyperparams],
):
    decorator = artifact_registry.register_artifact_hyperparams(artifact_type=artifact_type)
    result = decorator(hyperparams_class)
    assert result is hyperparams_class


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_registry, artifact_type",
    [(DummyScoreRegistry, DummyScoreType.DUMMY_SCORE_ARTIFACT)],
)
def test_register_artifact_hyperparams_duplicate(
    artifact_registry: Type[DummyArtifactRegistry[DummyScoreType, Score]],
    artifact_type: DummyScoreType,
):
    with pytest.warns(
        UserWarning,
        match=f"Hyperparams already registered for artifact_type={artifact_type.name}",
    ):
        decorator = artifact_registry.register_artifact_hyperparams(artifact_type=artifact_type)
        result = decorator(UnregisteredArtifactHyperparams)
        assert result is UnregisteredArtifactHyperparams


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_registry, artifact_type, artifact_class, resource_spec",
    [
        (
            AlternativeDummyScoreRegistry,
            "NEW_CUSTOM_1",
            NoHyperparamsArtifact,
            DummyResourceSpec(scale=1.0),
        ),
        (
            AlternativeDummyScoreRegistry,
            "NEW_CUSTOM_2",
            UnregisteredArtifact,
            DummyResourceSpec(scale=2.0),
        ),
    ],
)
def test_register_custom_artifact(
    artifact_registry: Type[DummyArtifactRegistry[DummyScoreType, Score]],
    artifact_type: str,
    artifact_class: Type[DummyArtifact[Any, ArtifactResult]],
    resource_spec: DummyResourceSpec,
):
    decorator = artifact_registry.register_custom_artifact(artifact_type=artifact_type)
    result = decorator(artifact_class)
    assert result is artifact_class
    artifact = artifact_registry.get(artifact_type=artifact_type, resource_spec=resource_spec)
    assert isinstance(artifact, artifact_class)
    assert artifact.resource_spec == resource_spec


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_registry, artifact_type",
    [
        (DummyScoreRegistry, "CUSTOM_SCORE_ARTIFACT"),
        (DummyScoreRegistry, "NO_HYPERPARAMS_CUSTOM_SCORE_ARTIFACT"),
    ],
)
def test_register_custom_artifact_duplicate(
    artifact_registry: Type[DummyArtifactRegistry[DummyScoreType, Score]],
    artifact_type: str,
):
    with pytest.warns(
        UserWarning,
        match=f"Artifact already registered for artifact_type={artifact_type}",
    ):
        decorator = artifact_registry.register_custom_artifact(artifact_type=artifact_type)
        result = decorator(UnregisteredArtifact)
        assert result is UnregisteredArtifact


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_registry, artifact_type, hyperparams_class",
    [
        (AlternativeDummyScoreRegistry, "NEW_CUSTOM_PARAMS_1", UnregisteredArtifactHyperparams),
        (AlternativeDummyScoreRegistry, "NEW_CUSTOM_PARAMS_2", DummyScoreHyperparams),
    ],
)
def test_register_custom_artifact_hyperparams(
    artifact_registry: Type[DummyArtifactRegistry[DummyScoreType, Score]],
    artifact_type: str,
    hyperparams_class: Type[ArtifactHyperparams],
):
    decorator = artifact_registry.register_custom_artifact_hyperparams(artifact_type=artifact_type)
    result = decorator(hyperparams_class)
    assert result is hyperparams_class


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_registry, artifact_type",
    [
        (DummyScoreRegistry, "CUSTOM_SCORE_ARTIFACT"),
    ],
)
def test_register_custom_artifact_hyperparams_duplicate(
    artifact_registry: Type[DummyArtifactRegistry[DummyScoreType, Score]],
    artifact_type: str,
):
    with pytest.warns(
        UserWarning,
        match=f"Hyperparams already registered for artifact_type={artifact_type}",
    ):
        decorator = artifact_registry.register_custom_artifact_hyperparams(
            artifact_type=artifact_type
        )
        result = decorator(UnregisteredArtifactHyperparams)
        assert result is UnregisteredArtifactHyperparams
