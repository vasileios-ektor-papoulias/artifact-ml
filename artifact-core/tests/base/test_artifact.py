import pytest

from tests.base.dummy.artifact_dependencies import DummyArtifactResources, DummyResourceSpec
from tests.base.dummy.artifacts import DummyScoreArtifact, DummyScoreHyperparams


@pytest.mark.unit
@pytest.mark.parametrize(
    "hyperparams, resource_spec, artifact_resources, expected",
    [
        (
            DummyScoreHyperparams(adjust_scale=True),
            DummyResourceSpec(scale=1),
            DummyArtifactResources(valid=True, x=1),
            1,
        ),
        (
            DummyScoreHyperparams(adjust_scale=True),
            DummyResourceSpec(scale=1),
            DummyArtifactResources(valid=False, x=1),
            1,
        ),
        (
            DummyScoreHyperparams(adjust_scale=True),
            DummyResourceSpec(scale=2),
            DummyArtifactResources(valid=True, x=1),
            2,
        ),
        (
            DummyScoreHyperparams(adjust_scale=True),
            DummyResourceSpec(scale=2),
            DummyArtifactResources(valid=True, x=2),
            4,
        ),
        (
            DummyScoreHyperparams(adjust_scale=False),
            DummyResourceSpec(scale=2),
            DummyArtifactResources(valid=True, x=1),
            1,
        ),
        (
            DummyScoreHyperparams(adjust_scale=False),
            DummyResourceSpec(scale=2),
            DummyArtifactResources(valid=True, x=2),
            2,
        ),
    ],
)
def test_compute(
    hyperparams: DummyScoreHyperparams,
    resource_spec: DummyResourceSpec,
    artifact_resources: DummyArtifactResources,
    expected: float,
):
    artifact = DummyScoreArtifact(resource_spec=resource_spec, hyperparams=hyperparams)
    if artifact_resources.valid:
        result = artifact.compute(resources=artifact_resources)
        assert result == expected
    else:
        with pytest.raises(ValueError, match="Invalid Resources"):
            artifact.compute(resources=artifact_resources)
