from typing import List, Tuple, Type, Union, cast

import numpy as np
import pytest
from artifact_core._libs.tools.schema.feature_spec.categorical import (
    CategoricalFeatureSpec,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtype, ls_categories",
    [
        (str, ["A", "B", "C"]),
        (str, ["cat", "dog"]),
        (np.str_, ["X", "Y", "Z"]),
        (object, ["one", "two", "three", "four"]),
    ],
)
def test_init(dtype: type, ls_categories: List[str]):
    spec = CategoricalFeatureSpec(dtype=dtype, ls_categories=ls_categories)
    assert spec.dtype is dtype
    assert spec.ls_categories == ls_categories
    assert spec.n_categories == len(ls_categories)


@pytest.mark.unit
def test_ls_categories_returns_copy(
    categorical_feature_spec: CategoricalFeatureSpec,
    categorical_categories: List[str],
):
    result = categorical_feature_spec.ls_categories
    result.append("D")
    assert categorical_feature_spec.ls_categories == categorical_categories


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_categories, expected_error, expected_match",
    [
        (("A", "B", "C"), TypeError, "must be a list"),
        (["A", 1, "C"], TypeError, "must be strings"),
        (["A", "B", "A"], ValueError, "duplicates"),
    ],
)
def test_init_validation_raises(
    invalid_categories: Union[Tuple[str, ...], List[Union[str, int]]],
    expected_error: Type[Exception],
    expected_match: str,
):
    with pytest.raises(expected_error, match=expected_match):
        CategoricalFeatureSpec(dtype=str, ls_categories=cast(List[str], invalid_categories))


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtype, ls_categories",
    [
        (str, ["A", "B", "C"]),
        (np.str_, ["X", "Y"]),
    ],
)
def test_to_dict(dtype: type, ls_categories: List[str]):
    spec = CategoricalFeatureSpec(dtype=dtype, ls_categories=ls_categories)
    result = spec.to_dict()
    assert "dtype" in result
    assert result["ls_categories"] == ls_categories


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtype_name, ls_categories",
    [
        ("str", ["A", "B", "C"]),
        ("numpy.str_", ["X", "Y"]),
    ],
)
def test_from_dict(dtype_name: str, ls_categories: List[str]):
    data = {"dtype": dtype_name, "ls_categories": ls_categories}
    spec = CategoricalFeatureSpec.from_dict(data=data)
    assert spec.ls_categories == ls_categories


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtype, ls_categories",
    [
        (str, ["A", "B", "C"]),
        (str, ["cat", "dog"]),
        (np.str_, ["X", "Y", "Z"]),
    ],
)
def test_to_dict_from_dict_roundtrip(dtype: type, ls_categories: List[str]):
    spec = CategoricalFeatureSpec(dtype=dtype, ls_categories=ls_categories)
    data = spec.to_dict()
    restored = CategoricalFeatureSpec.from_dict(data=data)
    assert restored == spec


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtype, ls_categories",
    [
        (str, ["A", "B", "C"]),
        (str, ["cat", "dog"]),
    ],
)
def test_serialization_roundtrip(dtype: type, ls_categories: List[str]):
    spec = CategoricalFeatureSpec(dtype=dtype, ls_categories=ls_categories)
    serialized = spec.serialize()
    restored = CategoricalFeatureSpec.deserialize(json_str=serialized)
    assert restored == spec


@pytest.mark.unit
@pytest.mark.parametrize(
    "spec_a_args, spec_b_args, expected",
    [
        ((str, ["A", "B"]), (str, ["A", "B"]), True),
        ((str, ["A", "B"]), (str, ["A", "B", "C"]), False),
        ((str, ["A", "B"]), (str, ["B", "A"]), False),
        ((str, ["A", "B"]), (np.str_, ["A", "B"]), False),
    ],
)
def test_equality(spec_a_args: tuple, spec_b_args: tuple, expected: bool):
    spec_a = CategoricalFeatureSpec(dtype=spec_a_args[0], ls_categories=spec_a_args[1])
    spec_b = CategoricalFeatureSpec(dtype=spec_b_args[0], ls_categories=spec_b_args[1])
    assert (spec_a == spec_b) == expected


@pytest.mark.unit
def test_equality_with_non_categorical_spec(categorical_feature_spec: CategoricalFeatureSpec):
    assert categorical_feature_spec.__eq__("not a spec") is NotImplemented
    assert categorical_feature_spec.__eq__(42) is NotImplemented
