from typing import List, Optional

import pytest
from artifact_core._libs.resource_specs.classification.spec import ClassSpec


@pytest.mark.unit
@pytest.mark.parametrize(
    "class_names, label_name, expected_label_name, expected_n_classes",
    [
        (["A", "B"], None, "label", 2),
        (["A", "B"], "target", "target", 2),
        (["A", "B", "C"], None, "label", 3),
        (["cat", "dog", "bird", "fish"], "animal", "animal", 4),
        (["0", "1"], "binary_label", "binary_label", 2),
    ],
)
def test_init(
    class_names: List[str],
    label_name: Optional[str],
    expected_label_name: str,
    expected_n_classes: int,
):
    spec = ClassSpec(class_names=class_names, label_name=label_name)
    assert spec.label_name == expected_label_name
    assert list(spec.class_names) == class_names
    assert spec.n_classes == expected_n_classes


@pytest.mark.unit
@pytest.mark.parametrize(
    "class_names, class_name, expected_idx",
    [
        (["A", "B", "C"], "A", 0),
        (["A", "B", "C"], "B", 1),
        (["A", "B", "C"], "C", 2),
        (["cat", "dog"], "dog", 1),
    ],
)
def test_get_class_idx(class_names: List[str], class_name: str, expected_idx: int):
    spec = ClassSpec(class_names=class_names)
    assert spec.get_class_idx(class_name) == expected_idx


@pytest.mark.unit
@pytest.mark.parametrize(
    "class_names, class_name",
    [
        (["A", "B", "C"], "D"),
        (["A", "B", "C"], "unknown"),
        (["cat", "dog"], "bird"),
    ],
)
def test_get_class_idx_raises_on_unknown_class(class_names: List[str], class_name: str):
    spec = ClassSpec(class_names=class_names)
    with pytest.raises(ValueError, match=f"Unknown class '{class_name}'"):
        spec.get_class_idx(class_name=class_name)


@pytest.mark.unit
@pytest.mark.parametrize(
    "class_names, class_name, expected",
    [
        (["A", "B", "C"], "A", True),
        (["A", "B", "C"], "B", True),
        (["A", "B", "C"], "D", False),
        (["cat", "dog"], "cat", True),
        (["cat", "dog"], "bird", False),
    ],
)
def test_has_class(class_names: List[str], class_name: str, expected: bool):
    spec = ClassSpec(class_names=class_names)
    assert spec.has_class(class_name=class_name) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "spec_a_args, spec_b_args, expected",
    [
        ((["A", "B"], "label"), (["A", "B"], "label"), True),
        ((["A", "B"], "label"), (["A", "B"], "target"), False),
        ((["A", "B"], "label"), (["B", "A"], "label"), False),
        ((["A", "B", "C"], "label"), (["A", "B"], "label"), False),
    ],
)
def test_equality(
    spec_a_args: tuple,
    spec_b_args: tuple,
    expected: bool,
):
    spec_a = ClassSpec(class_names=spec_a_args[0], label_name=spec_a_args[1])
    spec_b = ClassSpec(class_names=spec_b_args[0], label_name=spec_b_args[1])
    assert (spec_a == spec_b) == expected


@pytest.mark.unit
def test_equality_with_non_classspec(class_spec: ClassSpec):
    assert class_spec.__eq__("not a ClassSpec") == NotImplemented


@pytest.mark.unit
@pytest.mark.parametrize(
    "class_names, label_name",
    [
        (["A", "B"], "label"),
        (["A", "B", "C"], "target"),
        (["cat", "dog", "bird", "fish"], "animal"),
    ],
)
def test_serialization_roundtrip(class_names: List[str], label_name: str):
    spec = ClassSpec(class_names=class_names, label_name=label_name)
    json_str = spec.serialize()
    restored = ClassSpec.deserialize(json_str=json_str)
    assert restored == spec


@pytest.mark.unit
@pytest.mark.parametrize(
    "class_names, label_name",
    [
        (["A", "B"], "label"),
        (["A", "B", "C"], "target"),
    ],
)
def test_to_dict_from_dict_roundtrip(class_names: List[str], label_name: str):
    spec = ClassSpec(class_names=class_names, label_name=label_name)
    data = spec.to_dict()
    restored = ClassSpec.from_dict(data=data)
    assert restored == spec


@pytest.mark.unit
def test_repr(class_spec: ClassSpec, label_name: str, class_names: List[str]):
    repr_str = repr(class_spec)
    assert "ClassSpec" in repr_str
    assert label_name in repr_str
    assert str(len(class_names)) in repr_str
