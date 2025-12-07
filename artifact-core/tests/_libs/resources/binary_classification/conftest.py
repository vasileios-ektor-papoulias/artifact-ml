from dataclasses import dataclass
from typing import Tuple

import pytest
from artifact_core._libs.resource_specs.binary_classification.spec import BinaryClassSpec
from artifact_core._libs.resources.binary_classification.class_store import BinaryClassStore
from artifact_core._libs.resources.binary_classification.classification_results import (
    BinaryClassificationResults,
)
from artifact_core._libs.resources.binary_classification.distribution_store import (
    BinaryDistributionStore,
)


@dataclass(frozen=True)
class BinarySpecParams:
    label_name: str
    class_names: Tuple[str, str]
    positive_class: str
    negative_class: str


@pytest.fixture
def binary_spec_params() -> BinarySpecParams:
    return BinarySpecParams(
        label_name="target", class_names=("neg", "pos"), positive_class="pos", negative_class="neg"
    )


@pytest.fixture
def binary_spec(binary_spec_params: BinarySpecParams) -> BinaryClassSpec:
    return BinaryClassSpec(
        class_names=list(binary_spec_params.class_names),
        positive_class=binary_spec_params.positive_class,
        label_name=binary_spec_params.label_name,
    )


@pytest.fixture
def binary_class_store(binary_spec: BinaryClassSpec) -> BinaryClassStore:
    return BinaryClassStore.build_empty(class_spec=binary_spec)


@pytest.fixture
def binary_distribution_store(binary_spec: BinaryClassSpec) -> BinaryDistributionStore:
    return BinaryDistributionStore.build_empty(class_spec=binary_spec)


@pytest.fixture
def binary_classification_results(
    binary_spec: BinaryClassSpec,
) -> BinaryClassificationResults:
    return BinaryClassificationResults.build_empty(class_spec=binary_spec)
