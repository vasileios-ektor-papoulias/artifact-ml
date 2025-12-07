from typing import List, Optional

import pytest
from artifact_core._libs.resource_specs.binary_classification.spec import BinaryClassSpec


@pytest.mark.unit
@pytest.mark.parametrize(
    "class_names, positive_class, label_name, "
    "expected_label_name, expected_positive, expected_negative",
    [
        (["0", "1"], "1", None, "label", "1", "0"),
        (["0", "1"], "0", None, "label", "0", "1"),
        (["neg", "pos"], "pos", "target", "target", "pos", "neg"),
        (["neg", "pos"], "neg", "target", "target", "neg", "pos"),
        (["A", "B"], "A", "class", "class", "A", "B"),
        (["A", "B"], "B", "class", "class", "B", "A"),
    ],
)
def test_init(
    class_names: List[str],
    positive_class: str,
    label_name: Optional[str],
    expected_label_name: str,
    expected_positive: str,
    expected_negative: str,
):
    spec = BinaryClassSpec(
        class_names=class_names,
        positive_class=positive_class,
        label_name=label_name,
    )
    assert spec.label_name == expected_label_name
    assert list(spec.class_names) == class_names
    assert spec.n_classes == 2
    assert spec.positive_class == expected_positive
    assert spec.negative_class == expected_negative


@pytest.mark.unit
@pytest.mark.parametrize(
    "class_names, positive_class, expected_pos_idx, expected_neg_idx",
    [
        (["0", "1"], "1", 1, 0),
        (["0", "1"], "0", 0, 1),
        (["neg", "pos"], "pos", 1, 0),
        (["neg", "pos"], "neg", 0, 1),
    ],
)
def test_class_indices(
    class_names: List[str],
    positive_class: str,
    expected_pos_idx: int,
    expected_neg_idx: int,
):
    spec = BinaryClassSpec(class_names=class_names, positive_class=positive_class)
    assert spec.positive_class_idx == expected_pos_idx
    assert spec.negative_class_idx == expected_neg_idx


@pytest.mark.unit
@pytest.mark.parametrize(
    "class_names, error_match",
    [
        (["A"], "must contain exactly 2 categories"),
        (["A", "B", "C"], "must contain exactly 2 categories"),
        (["A", "A"], "must contain two distinct categories"),
    ],
)
def test_init_raises_on_invalid_class_count(class_names: List[str], error_match: str):
    with pytest.raises(ValueError, match=error_match):
        BinaryClassSpec(class_names=class_names, positive_class=class_names[0])


@pytest.mark.unit
def test_init_raises_on_invalid_positive_class():
    with pytest.raises(ValueError, match="positive_class.*not in"):
        BinaryClassSpec(class_names=["A", "B"], positive_class="C")


@pytest.mark.unit
@pytest.mark.parametrize(
    "class_names, positive_class, test_class, expected_positive, expected_negative",
    [
        (["0", "1"], "1", "1", True, False),
        (["0", "1"], "1", "0", False, True),
        (["neg", "pos"], "pos", "pos", True, False),
        (["neg", "pos"], "pos", "neg", False, True),
    ],
)
def test_is_positive_is_negative(
    class_names: List[str],
    positive_class: str,
    test_class: str,
    expected_positive: bool,
    expected_negative: bool,
):
    spec = BinaryClassSpec(class_names=class_names, positive_class=positive_class)
    assert spec.is_positive(test_class) == expected_positive
    assert spec.is_negative(test_class) == expected_negative


@pytest.mark.unit
@pytest.mark.parametrize("method", ["is_positive", "is_negative"])
def test_is_positive_is_negative_raises_on_unknown_class(method: str):
    spec = BinaryClassSpec(class_names=["A", "B"], positive_class="A")
    with pytest.raises(ValueError, match="Unknown class 'C'"):
        getattr(spec, method)("C")


@pytest.mark.unit
@pytest.mark.parametrize(
    "spec_a_args, spec_b_args, expected",
    [
        ((["A", "B"], "A", "label"), (["A", "B"], "A", "label"), True),
        ((["A", "B"], "A", "label"), (["A", "B"], "B", "label"), False),
        ((["A", "B"], "A", "label"), (["A", "B"], "A", "target"), False),
        ((["A", "B"], "A", "label"), (["B", "A"], "A", "label"), False),
    ],
)
def test_equality(
    spec_a_args: tuple,
    spec_b_args: tuple,
    expected: bool,
):
    spec_a = BinaryClassSpec(
        class_names=spec_a_args[0],
        positive_class=spec_a_args[1],
        label_name=spec_a_args[2],
    )
    spec_b = BinaryClassSpec(
        class_names=spec_b_args[0],
        positive_class=spec_b_args[1],
        label_name=spec_b_args[2],
    )
    assert (spec_a == spec_b) == expected


@pytest.mark.unit
def test_equality_with_non_binaryclassspec(binary_class_spec: BinaryClassSpec):
    assert binary_class_spec.__eq__("not a BinaryClassSpec") == NotImplemented


@pytest.mark.unit
@pytest.mark.parametrize(
    "class_names, positive_class, label_name",
    [
        (["0", "1"], "1", "label"),
        (["neg", "pos"], "pos", "target"),
        (["A", "B"], "A", "class"),
    ],
)
def test_serialization_roundtrip(class_names: List[str], positive_class: str, label_name: str):
    spec = BinaryClassSpec(
        class_names=class_names,
        positive_class=positive_class,
        label_name=label_name,
    )
    json_str = spec.serialize()
    restored = BinaryClassSpec.deserialize(json_str)
    assert restored == spec


@pytest.mark.unit
@pytest.mark.parametrize(
    "class_names, positive_class, label_name",
    [
        (["0", "1"], "1", "label"),
        (["neg", "pos"], "pos", "target"),
    ],
)
def test_to_dict_from_dict_roundtrip(class_names: List[str], positive_class: str, label_name: str):
    spec = BinaryClassSpec(
        class_names=class_names,
        positive_class=positive_class,
        label_name=label_name,
    )
    data = spec.to_dict()
    restored = BinaryClassSpec.from_dict(data)
    assert restored == spec


@pytest.mark.unit
def test_repr(
    binary_class_spec: BinaryClassSpec,
    label_name: str,
    positive_class: str,
    class_names: List[str],
):
    repr_str = repr(binary_class_spec)
    assert "BinaryClassSpec" in repr_str
    assert label_name in repr_str
    assert positive_class in repr_str
    negative_class = [c for c in class_names if c != positive_class][0]
    assert negative_class in repr_str


@pytest.mark.unit
def test_inherits_classspec_methods(
    binary_class_spec: BinaryClassSpec, class_names: List[str]
):
    assert binary_class_spec.has_class(class_names[0]) is True
    assert binary_class_spec.has_class("nonexistent") is False
    assert binary_class_spec.get_class_idx(class_names[0]) == 0
    assert binary_class_spec.get_class_idx(class_names[1]) == 1
