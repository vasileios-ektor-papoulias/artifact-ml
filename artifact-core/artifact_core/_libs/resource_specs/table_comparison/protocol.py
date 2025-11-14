from typing import Mapping, Protocol, Sequence

from artifact_core._libs.tools.schema.data_types.typing import TabularDataDType


class TabularDataSpecProtocol(Protocol):
    @property
    def features(self) -> Sequence[str]: ...

    @property
    def n_features(self) -> int: ...

    @property
    def cts_features(self) -> Sequence[str]: ...

    @property
    def n_cts_features(self) -> int: ...

    @property
    def cts_dtypes(self) -> Mapping[str, TabularDataDType]: ...

    @property
    def cat_features(self) -> Sequence[str]: ...

    @property
    def n_cat_features(self) -> int: ...

    @property
    def cat_dtypes(self) -> Mapping[str, TabularDataDType]: ...

    @property
    def cat_unique_map(self) -> Mapping[str, Sequence[str]]: ...

    @property
    def cat_unique_count_map(self) -> Mapping[str, int]: ...
