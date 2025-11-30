from typing import Union

import pytest
from artifact_core._base.orchestration.key_formatter import ArtifactKeyFormatter

from tests._base.dummy.types.scores import DummyScoreType


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, expected_key",
    [
        (DummyScoreType.DUMMY_SCORE_ARTIFACT, "DUMMY_SCORE_ARTIFACT"),
        (DummyScoreType.NO_HYPERPARAMS_ARTIFACT, "NO_HYPERPARAMS_ARTIFACT"),
        (DummyScoreType.IN_ALTERNATIVE_REGISTRY, "IN_ALTERNATIVE_REGISTRY"),
        (DummyScoreType.NOT_REGISTERED, "NOT_REGISTERED"),
        ("CUSTOM_SCORE_ARTIFACT", "CUSTOM_SCORE_ARTIFACT"),
        ("ANOTHER_ARTIFACT", "ANOTHER_ARTIFACT"),
        ("simple_string", "simple_string"),
        ("", ""),
        ("with-dashes", "with-dashes"),
        ("with_underscores", "with_underscores"),
        ("with.dots", "with.dots"),
        ("MixedCase", "MixedCase"),
        ("123numeric", "123numeric"),
    ],
)
def test_get_artifact_key(artifact_type: Union[DummyScoreType, str], expected_key: str):
    result = ArtifactKeyFormatter.get_artifact_key(artifact_type=artifact_type)
    assert result == expected_key
    assert isinstance(result, str)
