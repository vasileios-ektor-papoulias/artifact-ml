from typing import Any, Dict, List, Optional

import pandas as pd
import pytest
from artifact_core._libs.tools.schema.feature_partition.inference_engine import (
    FeaturePartitionInferenceEngine,
    FeaturePartitionStrategy,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "data, strategy, expected_cts, expected_cat",
    [
        (
            {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]},
            FeaturePartitionStrategy.NATIVE_PANDAS,
            ["a", "b"],
            [],
        ),
        (
            {"a": ["x", "y", "z"], "b": ["p", "q", "r"]},
            FeaturePartitionStrategy.NATIVE_PANDAS,
            [],
            ["a", "b"],
        ),
        (
            {"num": [1, 2, 3], "cat": ["a", "b", "c"]},
            FeaturePartitionStrategy.NATIVE_PANDAS,
            ["num"],
            ["cat"],
        ),
        (
            {"x": [1.0, 2.0], "y": [True, False], "z": ["a", "b"]},
            FeaturePartitionStrategy.NATIVE_PANDAS,
            ["x", "y"],
            ["z"],
        ),
        ({}, FeaturePartitionStrategy.NATIVE_PANDAS, [], []),
        ({"a": [1, 2, 3]}, None, ["a"], []),
        ({"a": ["x", "y"]}, None, [], ["a"]),
    ],
)
def test_infer(
    data: Dict[str, Any],
    strategy: Optional[FeaturePartitionStrategy],
    expected_cts: List[str],
    expected_cat: List[str],
):
    df = pd.DataFrame(data=data)
    if strategy is None:
        result = FeaturePartitionInferenceEngine.infer(df=df)
    else:
        result = FeaturePartitionInferenceEngine.infer(df=df, strategy=strategy)
    assert sorted(result.ls_cts_features) == sorted(expected_cts)
    assert sorted(result.ls_cat_features) == sorted(expected_cat)
    assert result.unknown_features == []
