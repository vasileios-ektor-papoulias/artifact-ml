from typing import Any, Callable, Type, TypeVar, Union

import pytest
from artifact_core._base.core.hyperparams import NO_ARTIFACT_HYPERPARAMS, NoArtifactHyperparams
from artifact_core._base.typing.artifact_result import ArtifactResult, Score

from tests._base.dummy.artifacts import (
    AlternativeRegistryArtifact,
    CustomScoreArtifact,
    CustomScoreHyperparams,
    DummyArtifact,
    DummyScoreArtifact,
    DummyScoreHyperparams,
    NoHyperparamsArtifact,
    NoHyperparamsCustomScoreArtifact,
    UnregisteredArtifact,
    UnregisteredArtifactHyperparams,
)
from tests._base.dummy.registries import (
    AlternativeDummyScoreRegistry,
    ArtifactRegistry,
    DummyScoreRegistry,
    DummyScoreType,
    InvalidParamDummyScoreRegistry,
    MissingParamDummyScoreRegistry,
)
from tests._base.dummy.resource_spec import DummyResourceSpec
from tests._base.dummy.resources import DummyArtifactResources


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_registry, artifact_type, resource_spec, expected_artifact_class, "
    + "expected_hyperparams, expect_raise_unregistered_artifact, "
    + "expect_raise_missing_config, expect_raise_missing_param",
    [
        (
            DummyScoreRegistry,
            DummyScoreType.DUMMY_SCORE_ARTIFACT,
            DummyResourceSpec(scale=1),
            DummyScoreArtifact,
            DummyScoreHyperparams(adjust_scale=True),
            False,
            False,
            False,
        ),
        (
            DummyScoreRegistry,
            DummyScoreType.DUMMY_SCORE_ARTIFACT,
            DummyResourceSpec(scale=10),
            DummyScoreArtifact,
            DummyScoreHyperparams(adjust_scale=True),
            False,
            False,
            False,
        ),
        (
            DummyScoreRegistry,
            DummyScoreType.NO_HYPERPARAMS_ARTIFACT,
            DummyResourceSpec(scale=1),
            NoHyperparamsArtifact,
            NO_ARTIFACT_HYPERPARAMS,
            False,
            False,
            False,
        ),
        (
            AlternativeDummyScoreRegistry,
            DummyScoreType.IN_ALTERNATIVE_REGISTRY,
            DummyResourceSpec(scale=1),
            AlternativeRegistryArtifact,
            NO_ARTIFACT_HYPERPARAMS,
            False,
            False,
            False,
        ),
        (
            DummyScoreRegistry,
            DummyScoreType.IN_ALTERNATIVE_REGISTRY,
            DummyResourceSpec(scale=1),
            AlternativeRegistryArtifact,
            None,
            True,
            False,
            False,
        ),
        (
            DummyScoreRegistry,
            DummyScoreType.NOT_REGISTERED,
            DummyResourceSpec(scale=1),
            AlternativeRegistryArtifact,
            None,
            True,
            False,
            False,
        ),
        (
            AlternativeDummyScoreRegistry,
            DummyScoreType.DUMMY_SCORE_ARTIFACT,
            DummyResourceSpec(scale=1),
            DummyScoreArtifact,
            DummyScoreHyperparams(adjust_scale=True),
            False,
            True,
            False,
        ),
        (
            MissingParamDummyScoreRegistry,
            DummyScoreType.DUMMY_SCORE_ARTIFACT,
            DummyResourceSpec(scale=1),
            DummyScoreArtifact,
            DummyScoreHyperparams(adjust_scale=True),
            False,
            False,
            True,
        ),
        (
            InvalidParamDummyScoreRegistry,
            DummyScoreType.DUMMY_SCORE_ARTIFACT,
            DummyResourceSpec(scale=1),
            DummyScoreArtifact,
            DummyScoreHyperparams(adjust_scale=True),
            False,
            False,
            True,
        ),
        (
            DummyScoreRegistry,
            "CUSTOM_SCORE_ARTIFACT",
            DummyResourceSpec(scale=1),
            CustomScoreArtifact,
            CustomScoreHyperparams(result=0),
            False,
            False,
            False,
        ),
        (
            DummyScoreRegistry,
            "NO_HYPERPARAMS_CUSTOM_SCORE_ARTIFACT",
            DummyResourceSpec(scale=1),
            NoHyperparamsCustomScoreArtifact,
            NO_ARTIFACT_HYPERPARAMS,
            False,
            False,
            False,
        ),
    ],
)
def test_get(
    artifact_registry: Type[
        ArtifactRegistry[DummyArtifactResources, DummyResourceSpec, DummyScoreType, Score]
    ],
    artifact_type: Union[DummyScoreType, str],
    resource_spec: DummyResourceSpec,
    expected_artifact_class: Type[DummyArtifact[Any, ArtifactResult]],
    expected_hyperparams: Union[DummyScoreHyperparams, NoArtifactHyperparams],
    expect_raise_unregistered_artifact: bool,
    expect_raise_missing_config: bool,
    expect_raise_missing_param: bool,
):
    str_artifact_type = artifact_type if isinstance(artifact_type, str) else artifact_type.name
    if expect_raise_unregistered_artifact:
        with pytest.raises(ValueError, match=f"Artifact {str_artifact_type} not registered"):
            artifact_registry.get(artifact_type=artifact_type, resource_spec=resource_spec)
    elif expect_raise_missing_config:
        with pytest.raises(
            ValueError,
            match="Missing config for hyperparams type " + f"{type(expected_hyperparams).__name__}",
        ):
            artifact_registry.get(artifact_type=artifact_type, resource_spec=resource_spec)
    elif expect_raise_missing_param:
        with pytest.raises(
            ValueError,
            match=f"Error instantiating '{type(expected_hyperparams).__name__}'",
        ):
            artifact_registry.get(artifact_type=artifact_type, resource_spec=resource_spec)
    else:
        artifact = artifact_registry.get(artifact_type=artifact_type, resource_spec=resource_spec)
        assert isinstance(artifact, expected_artifact_class)
        assert artifact.resource_spec == resource_spec
        assert artifact.hyperparams == expected_hyperparams


registreeT = TypeVar("registreeT")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("registry_method, artifact_type, registree, expected_warning_message"),
    [
        (
            DummyScoreRegistry.register_artifact,
            DummyScoreType.DUMMY_SCORE_ARTIFACT,
            UnregisteredArtifact,
            "Artifact already registered for artifact_type=DUMMY_SCORE_ARTIFACT",
        ),
        (
            DummyScoreRegistry.register_custom_artifact,
            "CUSTOM_SCORE_ARTIFACT",
            UnregisteredArtifact,
            "Artifact already registered for artifact_type=CUSTOM_SCORE_ARTIFACT",
        ),
        (
            DummyScoreRegistry.register_artifact_hyperparams,
            DummyScoreType.DUMMY_SCORE_ARTIFACT,
            UnregisteredArtifactHyperparams,
            "Hyperparams already registered for artifact_type=DUMMY_SCORE_ARTIFACT",
        ),
        (
            DummyScoreRegistry.register_custom_artifact_hyperparams,
            "CUSTOM_SCORE_ARTIFACT",
            UnregisteredArtifactHyperparams,
            "Hyperparams already registered for artifact_type=CUSTOM_SCORE_ARTIFACT",
        ),
    ],
)
def test_register_already_registered_artifact(
    registry_method: Callable[[Any], Callable[[registreeT], registreeT]],
    artifact_type: Union[DummyScoreType, str],
    registree: registreeT,
    expected_warning_message: str,
):
    with pytest.warns(UserWarning, match=expected_warning_message):
        registration_decorator = registry_method(artifact_type)
        result = registration_decorator(registree)
        assert result is registree
