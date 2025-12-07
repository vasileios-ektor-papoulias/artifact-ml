from typing import List

import pytest
from artifact_core._libs.tools.schema.feature_partition.feature_partition import (
    FeaturePartition,
    FeatureType,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "seq_all, seq_cts, seq_cat, expected_cts, expected_cat, expected_unknown",
    [
        (["a", "b", "c"], ["a"], ["b"], ["a"], ["b"], ["c"]),
        (["a", "b"], ["a", "b"], [], ["a", "b"], [], []),
        (["a", "b"], [], ["a", "b"], [], ["a", "b"], []),
        (["a", "b", "c"], [], [], [], [], ["a", "b", "c"]),
        (["x"], ["x"], [], ["x"], [], []),
        ([], [], [], [], [], []),
    ],
)
def test_build(
    seq_all: List[str],
    seq_cts: List[str],
    seq_cat: List[str],
    expected_cts: List[str],
    expected_cat: List[str],
    expected_unknown: List[str],
):
    partition = FeaturePartition.build(
        seq_all_features=seq_all, seq_cts_features=seq_cts, seq_cat_features=seq_cat
    )
    assert sorted(partition.ls_cts_features) == sorted(expected_cts)
    assert sorted(partition.ls_cat_features) == sorted(expected_cat)
    assert sorted(partition.unknown_features) == sorted(expected_unknown)


@pytest.mark.unit
@pytest.mark.parametrize(
    "seq_all, seq_cts, seq_cat, expected_match",
    [
        (["a", "b"], ["a", "c"], [], "not present"),
        (["a"], ["a", "b"], [], "not present"),
        (["a", "b"], [], ["c"], "not present"),
    ],
)
def test_build_missing_features_raises(
    seq_all: List[str], seq_cts: List[str], seq_cat: List[str], expected_match: str
):
    with pytest.raises(ValueError, match=expected_match):
        FeaturePartition.build(
            seq_all_features=seq_all, seq_cts_features=seq_cts, seq_cat_features=seq_cat
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "seq_all, seq_cts, seq_cat",
    [
        (["a", "b"], ["a"], ["a"]),
        (["x", "y", "z"], ["x", "y"], ["y", "z"]),
    ],
)
def test_build_overlapping_features_raises(
    seq_all: List[str], seq_cts: List[str], seq_cat: List[str]
):
    with pytest.raises(ValueError, match="both continuous and categorical"):
        FeaturePartition.build(
            seq_all_features=seq_all, seq_cts_features=seq_cts, seq_cat_features=seq_cat
        )


@pytest.mark.unit
def test_build_empty():
    partition = FeaturePartition.build_empty()
    assert partition.ls_all_features == []
    assert partition.ls_cts_features == []
    assert partition.ls_cat_features == []
    assert partition.unknown_features == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "features",
    [
        ["a", "b", "c"],
        ["x"],
        [],
    ],
)
def test_build_all_cts(features: List[str]):
    partition = FeaturePartition.build_all_cts(seq_all_features=features)
    assert sorted(partition.ls_cts_features) == sorted(features)
    assert partition.ls_cat_features == []
    assert partition.unknown_features == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "features",
    [
        ["a", "b", "c"],
        ["x"],
        [],
    ],
)
def test_build_all_cat(features: List[str]):
    partition = FeaturePartition.build_all_cat(seq_all_features=features)
    assert partition.ls_cts_features == []
    assert sorted(partition.ls_cat_features) == sorted(features)
    assert partition.unknown_features == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "features",
    [
        ["a", "b", "c"],
        ["x"],
        [],
    ],
)
def test_build_all_unknown(features: List[str]):
    partition = FeaturePartition.build_all_unknown(seq_all_features=features)
    assert partition.ls_cts_features == []
    assert partition.ls_cat_features == []
    assert sorted(partition.unknown_features) == sorted(features)


@pytest.mark.unit
def test_from_dict():
    dict_partition = {
        "a": FeatureType.CONTINUOUS,
        "b": FeatureType.CATEGORICAL,
        "c": FeatureType.UNKNOWN,
    }
    partition = FeaturePartition.from_dict(dict_partition=dict_partition)
    assert "a" in partition.ls_cts_features
    assert "b" in partition.ls_cat_features
    assert "c" in partition.unknown_features


@pytest.mark.unit
def test_dict_partition_property_returns_copy():
    partition = FeaturePartition.build(
        seq_all_features=["a", "b"], seq_cts_features=["a"], seq_cat_features=["b"]
    )
    dict_copy = partition.dict_partition
    dict_copy["c"] = FeatureType.UNKNOWN
    assert "c" not in partition.dict_partition


@pytest.mark.unit
@pytest.mark.parametrize(
    "seq_all, seq_cts, seq_cat",
    [
        (["a", "b", "c"], ["a"], ["b"]),
        (["x", "y"], ["x", "y"], []),
        (["p", "q"], [], ["p", "q"]),
    ],
)
def test_ls_all_features(seq_all: List[str], seq_cts: List[str], seq_cat: List[str]):
    partition = FeaturePartition.build(
        seq_all_features=seq_all, seq_cts_features=seq_cts, seq_cat_features=seq_cat
    )
    assert sorted(partition.ls_all_features) == sorted(seq_all)
