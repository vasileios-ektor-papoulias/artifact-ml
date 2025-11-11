from typing import Protocol

from artifact_core._libs.resource_spec.categorical.protocol import CategoricalFeatureSpecProtocol


class BinaryFeatureSpecProtocol(CategoricalFeatureSpecProtocol, Protocol):
    @property
    def positive_category(self) -> str: ...

    @property
    def negative_category(self) -> str: ...

    @property
    def positive_category_idx(self) -> int: ...

    @property
    def negative_category_idx(self) -> int: ...

    def is_positive(self, category: str) -> bool: ...

    def is_negative(self, category: str) -> bool: ...
