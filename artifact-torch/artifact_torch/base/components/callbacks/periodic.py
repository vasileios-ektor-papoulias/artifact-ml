from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

from artifact_experiment.base.callbacks.base import (
    Callback,
    CallbackResources,
)
from artifact_experiment.base.callbacks.cache import CacheCallback
from artifact_experiment.base.callbacks.tracking import TrackingCallback
from artifact_experiment.base.tracking.client import TrackingClient

CacheDataT = TypeVar("CacheDataT")
PeriodicCallbackResourcesT = TypeVar(
    "PeriodicCallbackResourcesT", bound="PeriodicCallbackResources"
)


@dataclass
class PeriodicCallbackResources(CallbackResources):
    step: int


class PeriodicActionTrigger:
    @staticmethod
    def should_trigger(step: int, period: int) -> bool:
        if period <= 0:
            return False
        return step % period == period - 1


class PeriodicCallback(
    Callback[PeriodicCallbackResourcesT],
    Generic[PeriodicCallbackResourcesT],
):
    def __init__(self, period: int):
        self._period = period

    @property
    def execution_interval(self) -> int:
        return self._period

    @abstractmethod
    def _execute(self, resources: PeriodicCallbackResourcesT): ...

    def execute(self, resources: PeriodicCallbackResourcesT):
        should_trigger = PeriodicActionTrigger.should_trigger(
            step=resources.step, period=self._period
        )
        if should_trigger:
            self._execute(resources=resources)


class PeriodicCacheCallback(
    CacheCallback[PeriodicCallbackResourcesT, CacheDataT],
    Generic[PeriodicCallbackResourcesT, CacheDataT],
):
    def __init__(self, key: str, period: int):
        super().__init__(key=key)
        self._period = period

    @abstractmethod
    def _compute(self, resources: PeriodicCallbackResourcesT) -> CacheDataT: ...

    @property
    def execution_interval(self) -> int:
        return self._period

    def execute(self, resources: PeriodicCallbackResourcesT):
        should_trigger = PeriodicActionTrigger.should_trigger(
            step=resources.step, period=self._period
        )
        if should_trigger:
            super().execute(resources=resources)


class PeriodicTrackingCallback(
    TrackingCallback[PeriodicCallbackResourcesT, CacheDataT],
    Generic[PeriodicCallbackResourcesT, CacheDataT],
):
    def __init__(self, key: str, period: int, tracking_client: Optional[TrackingClient] = None):
        super().__init__(key=key, tracking_client=tracking_client)
        self._period = period

    @abstractmethod
    def _compute(self, resources: PeriodicCallbackResourcesT) -> CacheDataT: ...

    @staticmethod
    @abstractmethod
    def _export(key: str, value: CacheDataT, tracking_client: TrackingClient): ...

    @property
    def execution_interval(self) -> int:
        return self._period

    def execute(self, resources: PeriodicCallbackResourcesT):
        should_trigger = PeriodicActionTrigger.should_trigger(
            step=resources.step, period=self._period
        )
        if should_trigger:
            super().execute(resources=resources)
