from typing import Any, Dict, List, Optional

import pandas as pd
import pytest
from artifact_core._libs.tools.schema.feature_partition.feature_partitioner import (
    FeaturePartitioner,
)
from artifact_core._libs.tools.schema.feature_partition.inference_engine import (
    FeaturePartitionStrategy,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "data, ls_cts, ls_cat, strategy, expected_cts, expected_cat",
    [
        (
            {"a": [1, 2], "b": ["x", "y"]},
            None,
            None,
            None,
            ["a"],
            ["b"],
        ),
        (
            {"a": [1, 2], "b": ["x", "y"]},
            ["a"],
            ["b"],
            None,
            ["a"],
            ["b"],
        ),
        (
            {"a": [1, 2], "b": [3, 4]},
            None,
            ["b"],
            None,
            ["a"],
            ["b"],
        ),
        (
            {"a": ["x", "y"], "b": ["p", "q"]},
            ["a"],
            None,
            None,
            ["a"],
            ["b"],
        ),
        (
            {"num_col": [1, 2, 3]},
            None,
            ["num_col"],
            None,
            [],
            ["num_col"],
        ),
        (
            {},
            None,
            None,
            None,
            [],
            [],
        ),
        (
            {"a": [1, 2], "b": ["x", "y"]},
            None,
            None,
            FeaturePartitionStrategy.NATIVE_PANDAS,
            ["a"],
            ["b"],
        ),
    ],
)
def test_partition_features(
    data: Dict[str, Any],
    ls_cts: Optional[List[str]],
    ls_cat: Optional[List[str]],
    strategy: Optional[FeaturePartitionStrategy],
    expected_cts: List[str],
    expected_cat: List[str],
):
    df = pd.DataFrame(data=data)
    if strategy is None:
        result = FeaturePartitioner.partition_features(
            df=df, ls_cts_features=ls_cts, ls_cat_features=ls_cat
        )
    else:
        result = FeaturePartitioner.partition_features(
            df=df, ls_cts_features=ls_cts, ls_cat_features=ls_cat, strategy=strategy
        )
    assert sorted(result.ls_cts_features) == sorted(expected_cts)
    assert sorted(result.ls_cat_features) == sorted(expected_cat)


@pytest.mark.unit
@pytest.mark.parametrize(
    "data, ls_cts, warn_on_missing, expected_warning_count, expected_warning_text",
    [
        (
            {"a": [1, 2]},
            ["a", "missing_col"],
            True,
            1,
            "Ignoring missing prescribed features: ['missing_col']",
        ),
        (
            {"a": [1, 2]},
            ["a", "missing_col"],
            False,
            0,
            None,
        ),
        (
            {"a": [1, 2]},
            ["a", "foo", "bar"],
            True,
            1,
            "Ignoring missing prescribed features: ['bar', 'foo']",
        ),
    ],
)
def test_partition_features_missing_warning(
    recwarn,
    data: Dict[str, Any],
    ls_cts: List[str],
    warn_on_missing: bool,
    expected_warning_count: int,
    expected_warning_text: Optional[str],
):
    df = pd.DataFrame(data=data)
    FeaturePartitioner.partition_features(
        df=df, ls_cts_features=ls_cts, warn_on_missing=warn_on_missing
    )
    assert len(recwarn) == expected_warning_count
    if expected_warning_text is not None:
        assert expected_warning_text in str(recwarn[0].message)
