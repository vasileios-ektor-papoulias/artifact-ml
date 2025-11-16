from typing import Protocol

from artifact_core._libs.resource_specs.classification.protocol import ClassSpecProtocol


class BinaryClassSpecProtocol(ClassSpecProtocol, Protocol):
    @property
    def positive_class(self) -> str: ...

    @property
    def negative_class(self) -> str: ...

    @property
    def positive_class_idx(self) -> int: ...

    @property
    def negative_class_idx(self) -> int: ...

    def is_positive(self, class_name: str) -> bool: ...

    def is_negative(self, class_name: str) -> bool: ...
