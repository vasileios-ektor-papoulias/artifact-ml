from typing import Mapping

import pytest
from artifact_core._libs.resource_specs.classification.spec import ClassSpec
from artifact_core._utils.collections.entity_store import IdentifierType


@pytest.fixture
def class_spec() -> ClassSpec:
    return ClassSpec(class_names=["A", "B", "C"], label_name="target")


@pytest.fixture
def binary_class_spec() -> ClassSpec:
    return ClassSpec(class_names=["neg", "pos"], label_name="label")


@pytest.fixture
def id_to_class_idx() -> Mapping[IdentifierType, int]:
    return {0: 0, 1: 1, 2: 2, 3: 0}


@pytest.fixture
def id_to_class() -> Mapping[IdentifierType, str]:
    return {0: "A", 1: "B", 2: "C", 3: "A"}
