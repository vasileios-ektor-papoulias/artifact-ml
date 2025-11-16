from dataclasses import dataclass

from artifact_experiment.spi.resources import (
    CacheCallbackResources,
    CallbackResources,
    TrackingCallbackResources,
)


@dataclass(frozen=True)
class PeriodicCallbackResources(CallbackResources):
    step: int


@dataclass(frozen=True)
class PeriodicCacheCallbackResources(CacheCallbackResources):
    step: int


@dataclass(frozen=True)
class PeriodicTrackingCallbackResources(TrackingCallbackResources):
    step: int
