from typing import Any, Type, Union

import pytest
from artifact_core.base.artifact_dependencies import (
    NO_ARTIFACT_HYPERPARAMS,
    ArtifactResult,
    NoArtifactHyperparams,
)

from tests.base.dummy.artifacts import (
    AlternativeRegistryArtifact,
    DummyArtifact,
    DummyScoreArtifact,
    DummyScoreHyperparams,
    NoHyperparamsArtifact,
)
from tests.base.dummy.registries import (
    AlternativeDummyScoreRegistry,
    ArtifactRegistry,
    DummyArtifactResources,
    DummyResourceSpec,
    DummyScoreRegistry,
    DummyScoreType,
    InvalidParamDummyScoreRegistry,
    MissingParamDummyScoreRegistry,
)


@pytest.mark.parametrize(
    "artifact_registry, artifact_type, resource_spec, expected_artifact_class, "
    + "expected_hyperparams, expect_raise_unregistered_artifact, expect_raise_missing_config, "
    + "expect_raise_missing_param",
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
    ],
)
def test_get(
    artifact_registry: Type[
        ArtifactRegistry[DummyScoreType, DummyArtifactResources, float, DummyResourceSpec]
    ],
    artifact_type: DummyScoreType,
    resource_spec: DummyResourceSpec,
    expected_artifact_class: Type[DummyArtifact[ArtifactResult, Any]],
    expected_hyperparams: Union[DummyScoreHyperparams, NoArtifactHyperparams],
    expect_raise_unregistered_artifact: bool,
    expect_raise_missing_config: bool,
    expect_raise_missing_param: bool,
):
    if expect_raise_unregistered_artifact:
        with pytest.raises(ValueError, match=f"Artifact {artifact_type.name} not registered"):
            artifact_registry.get(artifact_type=artifact_type, resource_spec=resource_spec)
    elif expect_raise_missing_config:
        with pytest.raises(
            ValueError,
            match=f"Missing config for hyperparams type {type(expected_hyperparams).__name__}",
        ):
            artifact_registry.get(artifact_type=artifact_type, resource_spec=resource_spec)
    elif expect_raise_missing_param:
        with pytest.raises(
            ValueError, match=f"Error instantiating '{type(expected_hyperparams).__name__}'"
        ):
            artifact_registry.get(artifact_type=artifact_type, resource_spec=resource_spec)
    else:
        artifact = artifact_registry.get(artifact_type=artifact_type, resource_spec=resource_spec)
        assert isinstance(artifact, expected_artifact_class)
        assert artifact.resource_spec == resource_spec
        assert artifact.hyperparams == expected_hyperparams
