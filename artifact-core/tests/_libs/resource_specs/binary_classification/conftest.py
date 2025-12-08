from typing import List

import pytest
from artifact_core._libs.resource_specs.binary_classification.spec import BinaryClassSpec


@pytest.fixture
def class_names() -> List[str]:
    return ["neg", "pos"]


@pytest.fixture
def positive_class() -> str:
    return "pos"


@pytest.fixture
def label_name() -> str:
    return "target"


@pytest.fixture
def binary_class_spec(
    class_names: List[str], positive_class: str, label_name: str
) -> BinaryClassSpec:
    return BinaryClassSpec(
        class_names=class_names, positive_class=positive_class, label_name=label_name
    )
