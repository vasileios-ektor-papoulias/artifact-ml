from typing import Dict, List, Protocol

from artifact_core._base.artifact_dependencies import ResourceSpecProtocol
from artifact_core._libs.resource_spec.tabular.types import TabularDataDType


class TabularDataSpecProtocol(ResourceSpecProtocol, Protocol):
    @property
    def ls_features(self) -> List[str]: ...

    @property
    def n_features(self) -> int: ...

    @property
    def ls_cts_features(self) -> List[str]: ...

    @property
    def n_cts_features(self) -> int: ...

    @property
    def dict_cts_dtypes(self) -> Dict[str, TabularDataDType]: ...

    @property
    def ls_cat_features(self) -> List[str]: ...

    @property
    def n_cat_features(self) -> int: ...

    @property
    def dict_cat_dtypes(self) -> Dict[str, TabularDataDType]: ...

    @property
    def cat_unique_map(self) -> Dict[str, List[str]]: ...

    @property
    def cat_unique_count_map(self) -> Dict[str, int]: ...
