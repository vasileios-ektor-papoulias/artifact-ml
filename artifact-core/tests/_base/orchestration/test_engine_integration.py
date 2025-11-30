import pytest

from tests._base.dummy.artifacts.scores.custom import CustomScoreArtifact, CustomScoreHyperparams
from tests._base.dummy.artifacts.scores.dummy import DummyScoreArtifact, DummyScoreHyperparams
from tests._base.dummy.artifacts.scores.no_hyperparams import NoHyperparamsArtifact
from tests._base.dummy.engine.engine import DummyArtifactEngine
from tests._base.dummy.registries.scores import DummyScoreRegistry
from tests._base.dummy.resource_spec import DummyResourceSpec
from tests._base.dummy.resources import DummyArtifactResources
from tests._base.dummy.types.scores import DummyScoreType


@pytest.fixture
def register_artifacts():
    DummyScoreRegistry.register_artifact_hyperparams(
        artifact_type=DummyScoreType.DUMMY_SCORE_ARTIFACT
    )(DummyScoreHyperparams)
    DummyScoreRegistry.register_artifact(artifact_type=DummyScoreType.DUMMY_SCORE_ARTIFACT)(
        DummyScoreArtifact
    )
    DummyScoreRegistry.register_artifact(artifact_type=DummyScoreType.NO_HYPERPARAMS_ARTIFACT)(
        NoHyperparamsArtifact
    )
    DummyScoreRegistry.register_custom_artifact_hyperparams(artifact_type="CUSTOM_SCORE_ARTIFACT")(
        CustomScoreHyperparams
    )
    DummyScoreRegistry.register_custom_artifact(artifact_type="CUSTOM_SCORE_ARTIFACT")(
        CustomScoreArtifact
    )
    yield


@pytest.mark.unit
@pytest.mark.parametrize(
    "score_type, resource_spec, resources, expected_result",
    [
        (
            DummyScoreType.DUMMY_SCORE_ARTIFACT,
            DummyResourceSpec(scale=1),
            DummyArtifactResources(valid=True, x=1),
            1,
        ),
        (
            DummyScoreType.DUMMY_SCORE_ARTIFACT,
            DummyResourceSpec(scale=2),
            DummyArtifactResources(valid=True, x=1),
            2,
        ),
        (
            DummyScoreType.DUMMY_SCORE_ARTIFACT,
            DummyResourceSpec(scale=2),
            DummyArtifactResources(valid=True, x=2),
            4,
        ),
        (
            DummyScoreType.NO_HYPERPARAMS_ARTIFACT,
            DummyResourceSpec(scale=1),
            DummyArtifactResources(valid=True, x=1),
            1,
        ),
        (
            DummyScoreType.NO_HYPERPARAMS_ARTIFACT,
            DummyResourceSpec(scale=2),
            DummyArtifactResources(valid=True, x=2),
            4,
        ),
        (
            "CUSTOM_SCORE_ARTIFACT",
            DummyResourceSpec(scale=1),
            DummyArtifactResources(valid=True, x=100),
            0,
        ),
    ],
)
def test_produce_score(
    register_artifacts,
    score_type: DummyScoreType,
    resource_spec: DummyResourceSpec,
    resources: DummyArtifactResources,
    expected_result: float,
):
    engine = DummyArtifactEngine.build(resource_spec=resource_spec)
    result = engine.produce_score(score_type=score_type, resources=resources)
    assert result == expected_result


@pytest.mark.unit
@pytest.mark.parametrize(
    "score_type, resource_spec, resources",
    [
        (
            DummyScoreType.DUMMY_SCORE_ARTIFACT,
            DummyResourceSpec(scale=1),
            DummyArtifactResources(valid=False, x=1),
        ),
        (
            DummyScoreType.NO_HYPERPARAMS_ARTIFACT,
            DummyResourceSpec(scale=2),
            DummyArtifactResources(valid=False, x=1),
        ),
    ],
)
def test_produce_score_invalid_resources(
    register_artifacts,
    score_type: DummyScoreType,
    resource_spec: DummyResourceSpec,
    resources: DummyArtifactResources,
):
    engine = DummyArtifactEngine.build(resource_spec=resource_spec)
    with pytest.raises(ValueError, match="Invalid Resources"):
        engine.produce_score(score_type=score_type, resources=resources)
