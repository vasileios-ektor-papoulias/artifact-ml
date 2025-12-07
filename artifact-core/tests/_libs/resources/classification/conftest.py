import pytest
from artifact_core._libs.resource_specs.classification.spec import ClassSpec
from artifact_core._libs.resources.classification.class_store import ClassStore
from artifact_core._libs.resources.classification.classification_results import (
    ClassificationResults,
)
from artifact_core._libs.resources.classification.distribution_store import ClassDistributionStore


@pytest.fixture
def class_spec() -> ClassSpec:
    return ClassSpec(class_names=["A", "B", "C"], label_name="target")


@pytest.fixture
def class_store(class_spec: ClassSpec) -> ClassStore:
    return ClassStore.build_empty(class_spec=class_spec)


@pytest.fixture
def distribution_store(class_spec: ClassSpec) -> ClassDistributionStore:
    return ClassDistributionStore.build_empty(class_spec=class_spec)


@pytest.fixture
def classification_results(class_spec: ClassSpec) -> ClassificationResults:
    return ClassificationResults.build_empty(class_spec=class_spec)
