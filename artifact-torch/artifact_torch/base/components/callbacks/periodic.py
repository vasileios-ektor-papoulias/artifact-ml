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

from artifact_torch.base.components.utils.periodic_action_trigger import PeriodicActionTrigger

PeriodicCallbackResourcesTContr = TypeVar(
    "PeriodicCallbackResourcesTContr", bound="PeriodicCallbackResources", contravariant=True
)


@dataclass
class PeriodicCallbackResources(CallbackResources):
    step: int


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
        should_trigger = PeriodicActionTrigger.should_trigger(
            step=resources.step, period=self._period
        )
        if should_trigger:
            self._execute(resources=resources)

    def should_trigger(self, step: int):
        return self._should_trigger(step=step)

    def _should_trigger(self, step: int):
        should_trigger = PeriodicActionTrigger.should_trigger(step=step, period=self._period)
        return should_trigger


CacheDataT = TypeVar("CacheDataT")


class PeriodicCacheCallback(
    CacheCallback[PeriodicCallbackResourcesTContr, CacheDataT],
    Generic[PeriodicCallbackResourcesTContr, CacheDataT],
):
    def __init__(self, key: str, period: int):
        super().__init__(key=key)
        self._period = period

    @abstractmethod
    def _compute(self, resources: PeriodicCallbackResourcesTContr) -> CacheDataT: ...

    @property
    def execution_interval(self) -> int:
        return self._period

    def execute(self, resources: PeriodicCallbackResourcesTContr):
        should_trigger = PeriodicActionTrigger.should_trigger(
            step=resources.step, period=self._period
        )
        if should_trigger:
            super().execute(resources=resources)

    def should_trigger(self, step: int):
        return self._should_trigger(step=step)

    def _should_trigger(self, step: int):
        should_trigger = PeriodicActionTrigger.should_trigger(step=step, period=self._period)
        return should_trigger


class PeriodicTrackingCallback(
    TrackingCallback[PeriodicCallbackResourcesTContr, CacheDataT],
    Generic[PeriodicCallbackResourcesTContr, CacheDataT],
):
    def __init__(self, key: str, period: int, tracking_client: Optional[TrackingClient] = None):
        super().__init__(key=key, tracking_client=tracking_client)
        self._period = period

    @abstractmethod
    def _compute(self, resources: PeriodicCallbackResourcesTContr) -> CacheDataT: ...

    @staticmethod
    @abstractmethod
    def _export(key: str, value: CacheDataT, tracking_client: TrackingClient): ...

    @property
    def execution_interval(self) -> int:
        return self._period

    def execute(self, resources: PeriodicCallbackResourcesTContr):
        should_trigger = PeriodicActionTrigger.should_trigger(
            step=resources.step, period=self._period
        )
        if should_trigger:
            super().execute(resources=resources)

    def should_trigger(self, step: int):
        return self._should_trigger(step=step)

    def _should_trigger(self, step: int):
        should_trigger = PeriodicActionTrigger.should_trigger(step=step, period=self._period)
        return should_trigger
