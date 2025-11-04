from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

from artifact_experiment.base.callbacks.base import (
    Callback,
    CallbackResources,
)
from artifact_experiment.base.callbacks.cache import CacheCallback, CacheCallbackResources
from artifact_experiment.base.callbacks.tracking import TrackingCallback, TrackingCallbackResources
from artifact_experiment.base.tracking.client import TrackingClient

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
CacheDataT = TypeVar("CacheDataT")


class PeriodicCacheCallback(
    CacheCallback[PeriodicCacheCallbackResourcesTContr, CacheDataT],
    Generic[PeriodicCacheCallbackResourcesTContr, CacheDataT],
):
    def __init__(self, base_key: str, period: int):
        super().__init__(base_key=base_key)
        self._period = period

    @abstractmethod
    def _compute(self, resources: PeriodicCacheCallbackResourcesTContr) -> CacheDataT: ...

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
CacheDataT = TypeVar("CacheDataT")


class PeriodicTrackingCallback(
    TrackingCallback[PeriodicTrackingCallbackResourcesTContr, CacheDataT],
    Generic[PeriodicTrackingCallbackResourcesTContr, CacheDataT],
):
    def __init__(
        self,
        base_key: str,
        period: int,
        tracking_client: Optional[TrackingClient] = None,
    ):
        super().__init__(base_key=base_key, tracking_client=tracking_client)
        self._period = period

    @abstractmethod
    def _compute(self, resources: PeriodicTrackingCallbackResourcesTContr) -> CacheDataT: ...

    @staticmethod
    @abstractmethod
    def _export(key: str, value: CacheDataT, tracking_client: TrackingClient): ...

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
