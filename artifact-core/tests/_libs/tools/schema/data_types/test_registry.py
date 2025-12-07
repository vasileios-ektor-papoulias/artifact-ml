import numpy as np
import pytest
from artifact_core._libs.tools.schema.data_types.registry import TABULAR_DATA_TYPE_REGISTRY


@pytest.mark.unit
def test_registry_is_dict():
    assert isinstance(TABULAR_DATA_TYPE_REGISTRY, dict)


@pytest.mark.unit
def test_registry_is_non_empty():
    assert len(TABULAR_DATA_TYPE_REGISTRY) > 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtype_name, expected_dtype",
    [
        ("int", int),
        ("float", float),
        ("str", str),
        ("bool", bool),
        ("numpy.int64", np.int64),
        ("numpy.float64", np.float64),
    ],
)
def test_registry_contains_common_types(dtype_name: str, expected_dtype: type):
    assert dtype_name in TABULAR_DATA_TYPE_REGISTRY
    assert TABULAR_DATA_TYPE_REGISTRY[dtype_name] is expected_dtype


@pytest.mark.unit
def test_registry_keys_are_strings():
    for key in TABULAR_DATA_TYPE_REGISTRY.keys():
        assert isinstance(key, str)
