from dataclasses import dataclass
from typing import Generic, TypeVar

from artifact_torch._base.components.callbacks.periodic import PeriodicTrackingCallbackResources
from artifact_torch._base.model.base import Model

ModelTCov = TypeVar("ModelTCov", bound=Model, covariant=True)


@dataclass(frozen=True)
class HookCallbackResources(PeriodicTrackingCallbackResources, Generic[ModelTCov]):
    model: ModelTCov
