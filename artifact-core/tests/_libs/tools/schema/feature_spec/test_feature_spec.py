import numpy as np
import pytest
from artifact_core._libs.tools.schema.feature_spec.feature_spec import FeatureSpec


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtype",
    [
        int,
        float,
        str,
        np.int64,
        np.float64,
    ],
)
def test_init(dtype: type):
    spec = FeatureSpec(dtype=dtype)
    assert spec.dtype is dtype


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtype, expected_dtype_name",
    [
        (int, "int"),
        (float, "float"),
        (str, "str"),
        (np.int64, "numpy.int64"),
        (np.float64, "numpy.float64"),
    ],
)
def test_to_dict(dtype: type, expected_dtype_name: str):
    spec = FeatureSpec(dtype=dtype)
    result = spec.to_dict()
    assert result == {"dtype": expected_dtype_name}


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtype_name, expected_dtype",
    [
        ("int", int),
        ("float", float),
        ("str", str),
        ("numpy.int64", np.int64),
        ("numpy.float64", np.float64),
    ],
)
def test_from_dict(dtype_name: str, expected_dtype: type):
    data = {"dtype": dtype_name}
    spec = FeatureSpec.from_dict(data=data)
    assert spec.dtype is expected_dtype


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtype",
    [
        int,
        float,
        str,
        np.int64,
        np.float64,
    ],
)
def test_to_dict_from_dict_roundtrip(dtype: type):
    spec = FeatureSpec(dtype=dtype)
    data = spec.to_dict()
    restored = FeatureSpec.from_dict(data=data)
    assert restored == spec


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtype",
    [
        int,
        float,
        str,
        np.int64,
        np.float64,
    ],
)
def test_serialization_roundtrip(dtype: type):
    spec = FeatureSpec(dtype=dtype)
    serialized = spec.serialize()
    restored = FeatureSpec.deserialize(json_str=serialized)
    assert restored == spec


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtype_a, dtype_b, expected",
    [
        (int, int, True),
        (float, float, True),
        (int, float, False),
        (np.int64, np.int64, True),
        (np.int64, np.int32, False),
    ],
)
def test_equality(dtype_a: type, dtype_b: type, expected: bool):
    spec_a = FeatureSpec(dtype=dtype_a)
    spec_b = FeatureSpec(dtype=dtype_b)
    assert (spec_a == spec_b) == expected


@pytest.mark.unit
def test_equality_with_non_feature_spec(feature_spec: FeatureSpec):
    assert feature_spec.__eq__("not a spec") is NotImplemented
    assert feature_spec.__eq__(42) is NotImplemented
    assert feature_spec.__eq__(None) is NotImplemented

