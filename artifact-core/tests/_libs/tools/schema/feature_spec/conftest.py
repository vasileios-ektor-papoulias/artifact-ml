from typing import List

import pytest
from artifact_core._libs.tools.schema.feature_spec.categorical import (
    CategoricalFeatureSpec,
)
from artifact_core._libs.tools.schema.feature_spec.continuous import (
    ContinuousFeatureSpec,
)
from artifact_core._libs.tools.schema.feature_spec.feature_spec import FeatureSpec


@pytest.fixture
def feature_spec() -> FeatureSpec:
    return FeatureSpec(dtype=float)


@pytest.fixture
def categorical_categories() -> List[str]:
    return ["A", "B", "C"]


@pytest.fixture
def categorical_feature_spec(categorical_categories: List[str]) -> CategoricalFeatureSpec:
    return CategoricalFeatureSpec(dtype=str, ls_categories=categorical_categories)


@pytest.fixture
def continuous_feature_spec() -> ContinuousFeatureSpec:
    return ContinuousFeatureSpec(dtype=float)
