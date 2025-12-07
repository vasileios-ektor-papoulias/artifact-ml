from typing import List

import pytest
from artifact_core._libs.resource_specs.classification.spec import ClassSpec
from artifact_core._libs.resources.classification.class_store import (  # noqa: E501
    ClassStore,
)
from artifact_core._libs.resources.classification.classification_results import (  # noqa: E501
    ClassificationResults,
)


@pytest.fixture
def class_names() -> List[str]:
    return ["A", "B", "C"]


@pytest.fixture
def class_spec(class_names: List[str]) -> ClassSpec:
    return ClassSpec(class_names=class_names, label_name="target")


@pytest.fixture
def true_class_store(class_spec: ClassSpec) -> ClassStore:
    return ClassStore.build_empty(class_spec=class_spec)


@pytest.fixture
def classification_results(class_spec: ClassSpec) -> ClassificationResults:
    return ClassificationResults.build_empty(class_spec=class_spec)
