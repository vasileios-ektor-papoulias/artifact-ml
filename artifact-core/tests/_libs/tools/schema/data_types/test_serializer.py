import numpy as np
import pytest
from artifact_core._libs.tools.schema.data_types.serializer import (
    TabularDataTypeSerializer,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtype_name, expected_dtype",
    [
        ("int", int),
        ("float", float),
        ("str", str),
        ("bool", bool),
        ("object", object),
        ("numpy.int64", np.int64),
        ("numpy.float64", np.float64),
        ("numpy.bool_", np.bool_),
        ("numpy.str_", np.str_),
    ],
)
def test_get_dtype(dtype_name: str, expected_dtype: type):
    result = TabularDataTypeSerializer.get_dtype(dtype_name=dtype_name)
    assert result is expected_dtype


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtype_name",
    [
        "unknown_type",
        "InvalidDtype",
        "numpy.invalid",
        "",
    ],
)
def test_get_dtype_unknown_raises(dtype_name: str):
    with pytest.raises(ValueError, match="Unknown dtype"):
        TabularDataTypeSerializer.get_dtype(dtype_name=dtype_name)


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtype, expected_name",
    [
        (int, "int"),
        (float, "float"),
        (str, "str"),
        (bool, "bool"),
        (object, "object"),
        (np.int64, "numpy.int64"),
        (np.float64, "numpy.float64"),
        (np.bool_, "numpy.bool"),
        (np.str_, "numpy.str_"),
    ],
)
def test_get_dtype_name(dtype: type, expected_name: str):
    result = TabularDataTypeSerializer.get_dtype_name(dtype=dtype)
    assert result == expected_name


@pytest.mark.unit
def test_get_dtype_name_unsupported_raises():
    class CustomType:
        pass

    with pytest.raises(ValueError, match="Unsupported dtype"):
        TabularDataTypeSerializer.get_dtype_name(dtype=CustomType)


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtype_name",
    [
        "int",
        "float",
        "str",
        "numpy.int64",
        "numpy.float64",
    ],
)
def test_roundtrip(dtype_name: str):
    dtype = TabularDataTypeSerializer.get_dtype(dtype_name=dtype_name)
    restored_name = TabularDataTypeSerializer.get_dtype_name(dtype=dtype)
    assert restored_name == dtype_name
