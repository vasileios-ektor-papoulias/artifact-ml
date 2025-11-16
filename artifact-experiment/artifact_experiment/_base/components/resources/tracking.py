from dataclasses import KW_ONLY, dataclass
from typing import Optional

from artifact_experiment._base.components.resources.cache import CacheCallbackResources
from artifact_experiment._base.primitives.data_split import DataSplit


@dataclass(frozen=True)
class TrackingCallbackResources(CacheCallbackResources):
    _: KW_ONLY
    data_split: Optional[DataSplit] = None
