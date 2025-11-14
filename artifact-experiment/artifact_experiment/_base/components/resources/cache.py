from dataclasses import KW_ONLY, dataclass
from typing import Optional

from artifact_experiment._base.components.resources.base import CallbackResources


@dataclass(frozen=True)
class CacheCallbackResources(CallbackResources):
    _: KW_ONLY
    trigger: Optional[str] = None
