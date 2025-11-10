from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

from artifact_experiment.base.components.callbacks.base import Callback, CallbackResources
from artifact_experiment.base.components.callbacks.cache import (
    CacheCallback,
    CacheCallbackResources,
)
from artifact_experiment.base.components.callbacks.tracking import (
    TrackingCallback,
    TrackingCallbackResources,
)
from artifact_experiment.base.tracking.background.writer import TrackingData, TrackingQueueWriter

from artifact_torch.base.components.utils.periodic_action_trigger import PeriodicActionTrigger


@dataclass(frozen=True)
class PeriodicCallbackResources(CallbackResources):
    step: int


PeriodicCallbackResourcesTContr = TypeVar(
    "PeriodicCallbackResourcesTContr", bound=PeriodicCallbackResources, contravariant=True
)


class PeriodicCallback(
    Callback[PeriodicCallbackResourcesTContr],
    Generic[PeriodicCallbackResourcesTContr],
):
    def __init__(self, period: int):
        self._period = period

    @property
    def execution_interval(self) -> int:
        return self._period

    @abstractmethod
    def _execute(self, resources: PeriodicCallbackResourcesTContr): ...

    def execute(self, resources: PeriodicCallbackResourcesTContr):
        should_trigger = self._should_trigger(step=resources.step)
        if should_trigger:
            self._execute(resources=resources)

    def _should_trigger(self, step: int):
        should_trigger = PeriodicActionTrigger.should_trigger(step=step, period=self._period)
        return should_trigger


@dataclass(frozen=True)
class PeriodicCacheCallbackResources(CacheCallbackResources):
    step: int


PeriodicCacheCallbackResourcesTContr = TypeVar(
    "PeriodicCacheCallbackResourcesTContr", bound=PeriodicCacheCallbackResources, contravariant=True
)
CacheDataTCov = TypeVar("CacheDataTCov", bound=TrackingData, covariant=True)


class PeriodicCacheCallback(
    CacheCallback[PeriodicCacheCallbackResourcesTContr, CacheDataTCov],
    Generic[PeriodicCacheCallbackResourcesTContr, CacheDataTCov],
):
    def __init__(self, base_key: str, period: int):
        super().__init__(base_key=base_key)
        self._period = period

    @abstractmethod
    def _compute(self, resources: PeriodicCacheCallbackResourcesTContr) -> CacheDataTCov: ...

    @property
    def execution_interval(self) -> int:
        return self._period

    def execute(self, resources: PeriodicCacheCallbackResourcesTContr):
        self._clear()
        should_trigger = self._should_trigger(step=resources.step)
        if should_trigger:
            super().execute(resources=resources)

    def _should_trigger(self, step: int):
        should_trigger = PeriodicActionTrigger.should_trigger(step=step, period=self._period)
        return should_trigger


@dataclass(frozen=True)
class PeriodicTrackingCallbackResources(TrackingCallbackResources):
    step: int


PeriodicTrackingCallbackResourcesTContr = TypeVar(
    "PeriodicTrackingCallbackResourcesTContr",
    bound=PeriodicTrackingCallbackResources,
    contravariant=True,
)
CacheDataTCov = TypeVar("CacheDataTCov", bound=TrackingData, covariant=True)


class PeriodicTrackingCallback(
    TrackingCallback[PeriodicTrackingCallbackResourcesTContr, CacheDataTCov],
    Generic[PeriodicTrackingCallbackResourcesTContr, CacheDataTCov],
):
    def __init__(
        self,
        base_key: str,
        period: int,
        writer: Optional[TrackingQueueWriter[CacheDataTCov]] = None,
    ):
        super().__init__(base_key=base_key, writer=writer)
        self._period = period

    @abstractmethod
    def _compute(self, resources: PeriodicTrackingCallbackResourcesTContr) -> CacheDataTCov: ...

    @property
    def execution_interval(self) -> int:
        return self._period

    def execute(self, resources: PeriodicTrackingCallbackResourcesTContr):
        self._clear()
        should_trigger = self._should_trigger(step=resources.step)
        if should_trigger:
            super().execute(resources=resources)

    def _should_trigger(self, step: int):
        should_trigger = PeriodicActionTrigger.should_trigger(step=step, period=self._period)
        return should_trigger
