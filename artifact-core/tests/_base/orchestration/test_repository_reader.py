from typing import Any, Dict, Mapping, Type, Union

import pytest
from artifact_core._base.core.artifact import Artifact
from artifact_core._base.core.hyperparams import (
    NO_ARTIFACT_HYPERPARAMS,
    ArtifactHyperparams,
    NoArtifactHyperparams,
)
from artifact_core._base.orchestration.repository import (
    ArtifactHyperparamsRepository,
    ArtifactRepository,
)
from artifact_core._base.orchestration.repository_reader import ArtifactRepositoryReader
from artifact_core._base.typing.artifact_result import Score

from tests._base.dummy.artifacts.scores.custom import CustomScoreArtifact, CustomScoreHyperparams
from tests._base.dummy.artifacts.scores.dummy import DummyScoreArtifact, DummyScoreHyperparams
from tests._base.dummy.artifacts.scores.no_hyperparams import NoHyperparamsArtifact
from tests._base.dummy.resource_spec import DummyResourceSpec
from tests._base.dummy.resources import DummyArtifactResources
from tests._base.dummy.types.scores import DummyScoreType


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, resource_spec, artifact_repository, hyperparams_repository, "
    + "artifact_configurations, expected_artifact_class, expected_hyperparams",
    [
        (
            DummyScoreType.DUMMY_SCORE_ARTIFACT,
            DummyResourceSpec(scale=1.0),
            {"DUMMY_SCORE_ARTIFACT": DummyScoreArtifact},
            {"DUMMY_SCORE_ARTIFACT": DummyScoreHyperparams},
            {"DUMMY_SCORE_ARTIFACT": {"adjust_scale": True}},
            DummyScoreArtifact,
            DummyScoreHyperparams(adjust_scale=True),
        ),
        (
            DummyScoreType.DUMMY_SCORE_ARTIFACT,
            DummyResourceSpec(scale=10.0),
            {"DUMMY_SCORE_ARTIFACT": DummyScoreArtifact},
            {"DUMMY_SCORE_ARTIFACT": DummyScoreHyperparams},
            {"DUMMY_SCORE_ARTIFACT": {"adjust_scale": False}},
            DummyScoreArtifact,
            DummyScoreHyperparams(adjust_scale=False),
        ),
        (
            DummyScoreType.NO_HYPERPARAMS_ARTIFACT,
            DummyResourceSpec(scale=1.0),
            {"NO_HYPERPARAMS_ARTIFACT": NoHyperparamsArtifact},
            {},
            {},
            NoHyperparamsArtifact,
            NO_ARTIFACT_HYPERPARAMS,
        ),
        (
            DummyScoreType.NO_HYPERPARAMS_ARTIFACT,
            DummyResourceSpec(scale=2.0),
            {"NO_HYPERPARAMS_ARTIFACT": NoHyperparamsArtifact},
            {"NO_HYPERPARAMS_ARTIFACT": NoArtifactHyperparams},
            {},
            NoHyperparamsArtifact,
            NO_ARTIFACT_HYPERPARAMS,
        ),
        (
            "CUSTOM_SCORE_ARTIFACT",
            DummyResourceSpec(scale=1.0),
            {"CUSTOM_SCORE_ARTIFACT": CustomScoreArtifact},
            {"CUSTOM_SCORE_ARTIFACT": CustomScoreHyperparams},
            {"CUSTOM_SCORE_ARTIFACT": {"result": 42.0}},
            CustomScoreArtifact,
            CustomScoreHyperparams(result=42.0),
        ),
        (
            "CUSTOM_SCORE_ARTIFACT",
            DummyResourceSpec(scale=5.0),
            {"CUSTOM_SCORE_ARTIFACT": CustomScoreArtifact},
            {"CUSTOM_SCORE_ARTIFACT": CustomScoreHyperparams},
            {"CUSTOM_SCORE_ARTIFACT": {"result": 0.0}},
            CustomScoreArtifact,
            CustomScoreHyperparams(result=0.0),
        ),
        (
            DummyScoreType.DUMMY_SCORE_ARTIFACT,
            DummyResourceSpec(scale=3.0),
            {
                "DUMMY_SCORE_ARTIFACT": DummyScoreArtifact,
                "OTHER_ARTIFACT": CustomScoreArtifact,
            },
            {
                "DUMMY_SCORE_ARTIFACT": DummyScoreHyperparams,
                "OTHER_ARTIFACT": CustomScoreHyperparams,
            },
            {
                "DUMMY_SCORE_ARTIFACT": {"adjust_scale": True},
                "OTHER_ARTIFACT": {"result": 1.0},
            },
            DummyScoreArtifact,
            DummyScoreHyperparams(adjust_scale=True),
        ),
    ],
)
def test_get(
    artifact_type: Union[DummyScoreType, str],
    resource_spec: DummyResourceSpec,
    artifact_repository: ArtifactRepository[DummyArtifactResources, DummyResourceSpec, Score],
    hyperparams_repository: ArtifactHyperparamsRepository,
    artifact_configurations: Mapping[str, Mapping[str, Any]],
    expected_artifact_class: Type[Artifact[Any, Any, Any, Any]],
    expected_hyperparams: Union[ArtifactHyperparams, NoArtifactHyperparams],
):
    artifact = ArtifactRepositoryReader.get(
        artifact_type=artifact_type,
        resource_spec=resource_spec,
        artifact_repository=artifact_repository,
        artifact_hyperparams_repository=hyperparams_repository,
        artifact_configurations=artifact_configurations,
    )

    assert isinstance(artifact, expected_artifact_class)
    assert artifact.resource_spec == resource_spec
    assert artifact.hyperparams == expected_hyperparams


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, expected_error_message",
    [
        (DummyScoreType.NOT_REGISTERED, "Artifact NOT_REGISTERED not registered"),
        ("UNREGISTERED_ARTIFACT", "Artifact UNREGISTERED_ARTIFACT not registered"),
        ("", "Artifact  not registered"),
    ],
)
def test_get_unregistered_artifact(
    artifact_type: Union[DummyScoreType, str],
    expected_error_message: str,
):
    resource_spec = DummyResourceSpec(scale=1.0)
    artifact_repository: ArtifactRepository[DummyArtifactResources, DummyResourceSpec, Score] = {}
    hyperparams_repository: ArtifactHyperparamsRepository = {}
    artifact_configurations: Dict[str, Dict[str, Any]] = {}

    with pytest.raises(ValueError, match=expected_error_message):
        ArtifactRepositoryReader.get(
            artifact_type=artifact_type,
            resource_spec=resource_spec,
            artifact_repository=artifact_repository,
            artifact_hyperparams_repository=hyperparams_repository,
            artifact_configurations=artifact_configurations,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, hyperparams_class",
    [
        (DummyScoreType.DUMMY_SCORE_ARTIFACT, DummyScoreHyperparams),
        ("CUSTOM_SCORE_ARTIFACT", CustomScoreHyperparams),
    ],
)
def test_get_missing_config(
    artifact_type: Union[DummyScoreType, str],
    hyperparams_class: Type[ArtifactHyperparams],
):
    key = artifact_type if isinstance(artifact_type, str) else artifact_type.name
    resource_spec = DummyResourceSpec(scale=1.0)
    artifact_repository: ArtifactRepository[DummyArtifactResources, DummyResourceSpec, Score] = {
        key: DummyScoreArtifact
    }
    hyperparams_repository: ArtifactHyperparamsRepository = {key: hyperparams_class}
    artifact_configurations: Dict[str, Dict[str, Any]] = {}

    with pytest.raises(
        ValueError,
        match=f"Missing config for hyperparams type {hyperparams_class.__name__}",
    ):
        ArtifactRepositoryReader.get(
            artifact_type=artifact_type,
            resource_spec=resource_spec,
            artifact_repository=artifact_repository,
            artifact_hyperparams_repository=hyperparams_repository,
            artifact_configurations=artifact_configurations,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, hyperparams_class, invalid_config",
    [
        (DummyScoreType.DUMMY_SCORE_ARTIFACT, DummyScoreHyperparams, {}),
        (DummyScoreType.DUMMY_SCORE_ARTIFACT, DummyScoreHyperparams, {"invalid_param": True}),
        ("CUSTOM_SCORE_ARTIFACT", CustomScoreHyperparams, {}),
        ("CUSTOM_SCORE_ARTIFACT", CustomScoreHyperparams, {"wrong_field": 0}),
    ],
)
def test_get_invalid_config(
    artifact_type: Union[DummyScoreType, str],
    hyperparams_class: Type[ArtifactHyperparams],
    invalid_config: Dict[str, Any],
):
    key = artifact_type if isinstance(artifact_type, str) else artifact_type.name
    resource_spec = DummyResourceSpec(scale=1.0)
    artifact_repository: ArtifactRepository[DummyArtifactResources, DummyResourceSpec, Score] = {
        key: DummyScoreArtifact
    }
    hyperparams_repository: ArtifactHyperparamsRepository = {key: hyperparams_class}
    artifact_configurations = {key: invalid_config}

    with pytest.raises(
        ValueError,
        match=f"Error instantiating '{hyperparams_class.__name__}'",
    ):
        ArtifactRepositoryReader.get(
            artifact_type=artifact_type,
            resource_spec=resource_spec,
            artifact_repository=artifact_repository,
            artifact_hyperparams_repository=hyperparams_repository,
            artifact_configurations=artifact_configurations,
        )
