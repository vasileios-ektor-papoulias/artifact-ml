import numpy as np
import pytest
from artifact_core._libs.tools.schema.feature_spec.continuous import (
    ContinuousFeatureSpec,
)
from artifact_core._libs.tools.schema.feature_spec.feature_spec import FeatureSpec


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtype",
    [
        int,
        float,
        np.int64,
        np.float64,
    ],
)
def test_init(dtype: type):
    spec = ContinuousFeatureSpec(dtype=dtype)
    assert spec.dtype is dtype


@pytest.mark.unit
def test_inherits_from_feature_spec():
    spec = ContinuousFeatureSpec(dtype=float)
    assert isinstance(spec, FeatureSpec)


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtype",
    [
        int,
        float,
        np.int64,
        np.float64,
    ],
)
def test_to_dict_from_dict_roundtrip(dtype: type):
    spec = ContinuousFeatureSpec(dtype=dtype)
    data = spec.to_dict()
    restored = ContinuousFeatureSpec.from_dict(data=data)
    assert restored.dtype is dtype


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtype",
    [
        int,
        float,
        np.float64,
    ],
)
def test_serialization_roundtrip(dtype: type):
    spec = ContinuousFeatureSpec(dtype=dtype)
    serialized = spec.serialize()
    restored = ContinuousFeatureSpec.deserialize(json_str=serialized)
    assert restored.dtype is dtype
