from typing import Protocol

from artifact_core.libs.resource_spec.categorical.protocol import CategoricalFeatureSpecProtocol


class BinaryFeatureSpecProtocol(CategoricalFeatureSpecProtocol, Protocol):
    @property
    def true_category(self) -> str: ...

    @property
    def false_category(self) -> str: ...

    def is_positive(self, category: str) -> bool: ...

    def is_negative(self, category: str) -> bool: ...
