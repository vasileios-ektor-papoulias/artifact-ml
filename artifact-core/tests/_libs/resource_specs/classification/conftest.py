from typing import List

import pytest
from artifact_core._libs.resource_specs.classification.spec import ClassSpec


@pytest.fixture
def class_names() -> List[str]:
    return ["A", "B", "C"]


@pytest.fixture
def label_name() -> str:
    return "target"


@pytest.fixture
def class_spec(class_names: List[str], label_name: str) -> ClassSpec:
    return ClassSpec(class_names=class_names, label_name=label_name)
