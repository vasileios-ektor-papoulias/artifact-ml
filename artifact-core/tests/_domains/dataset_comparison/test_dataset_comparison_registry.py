from typing import Type, Union

import pytest
from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._base.typing.artifact_result import Score

from tests._domains.dataset_comparison.dummy.artifacts.scores.dummy import (
    DummyDatasetComparisonScore,
    DummyDatasetComparisonScoreHyperparams,
)
from tests._domains.dataset_comparison.dummy.registries.base import DummyDatasetComparisonRegistry
from tests._domains.dataset_comparison.dummy.registries.scores import (
    DummyDatasetComparisonScoreRegistry,
)
from tests._domains.dataset_comparison.dummy.resource_spec import DummyDatasetSpec
from tests._domains.dataset_comparison.dummy.types.scores import DummyDatasetComparisonScoreType


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_registry, artifact_type, resource_spec, "
    + "expected_artifact_class, expected_hyperparams",
    [
        (
            DummyDatasetComparisonScoreRegistry,
            DummyDatasetComparisonScoreType.DUMMY_SCORE,
            DummyDatasetSpec(scale=1),
            DummyDatasetComparisonScore,
            DummyDatasetComparisonScoreHyperparams(adjust_scale=True),
        ),
        (
            DummyDatasetComparisonScoreRegistry,
            DummyDatasetComparisonScoreType.DUMMY_SCORE,
            DummyDatasetSpec(scale=10),
            DummyDatasetComparisonScore,
            DummyDatasetComparisonScoreHyperparams(adjust_scale=True),
        ),
    ],
)
def test_get(
    artifact_registry: Type[DummyDatasetComparisonRegistry[DummyDatasetComparisonScoreType, Score]],
    artifact_type: Union[DummyDatasetComparisonScoreType, str],
    resource_spec: DummyDatasetSpec,
    expected_artifact_class: Type,
    expected_hyperparams: ArtifactHyperparams,
):
    artifact = artifact_registry.get(artifact_type=artifact_type, resource_spec=resource_spec)
    assert isinstance(artifact, expected_artifact_class)
    assert artifact.resource_spec == resource_spec
    assert artifact.hyperparams == expected_hyperparams
