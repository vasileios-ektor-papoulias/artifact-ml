from typing import TypeVar

from artifact_core._libs.tools.schema.feature_spec.feature_spec import FeatureSpec

ContinuousFeatureSpecT = TypeVar("ContinuousFeatureSpecT", bound="ContinuousFeatureSpec")


class ContinuousFeatureSpec(FeatureSpec):
    pass
