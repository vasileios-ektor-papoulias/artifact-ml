from typing import Any, Dict, Generic, TypeVar

from artifact_experiment.base.components.callbacks.tracking import (
    TrackingCallback,
    TrackingCallbackResources,
)
from artifact_experiment.base.components.handlers.cache import CacheCallbackHandler
from artifact_experiment.base.entities.tracking_data import TrackingData

TrackingCallbackTCov = TypeVar(
    "TrackingCallbackTCov", bound=TrackingCallback[Any, Any], covariant=True
)
TrackingCallbackResourcesTContr = TypeVar(
    "TrackingCallbackResourcesTContr", bound=TrackingCallbackResources, contravariant=True
)
CacheDataTCov = TypeVar("CacheDataTCov", bound=TrackingData, covariant=True)


class TrackingCallbackHandler(
    CacheCallbackHandler[TrackingCallbackTCov, TrackingCallbackResourcesTContr, CacheDataTCov],
    Generic[TrackingCallbackTCov, TrackingCallbackResourcesTContr, CacheDataTCov],
):
    pass


TrackingScoreHandler = TrackingCallbackHandler[
    TrackingCallbackTCov, TrackingCallbackResourcesTContr, float
]
TrackingArrayHandler = TrackingCallbackHandler[
    TrackingCallbackTCov, TrackingCallbackResourcesTContr, Array
]
TrackingPlotHandler = TrackingCallbackHandler[
    TrackingCallbackTCov, TrackingCallbackResourcesTContr, Figure
]
TrackingScoreCollectionHandler = TrackingCallbackHandler[
    TrackingCallbackTCov, TrackingCallbackResourcesTContr, Dict[str, float]
]
TrackingArrayCollectionHandler = TrackingCallbackHandler[
    TrackingCallbackTCov, TrackingCallbackResourcesTContr, Dict[str, Array]
]
TrackingPlotCollectionHandler = TrackingCallbackHandler[
    TrackingCallbackTCov, TrackingCallbackResourcesTContr, Dict[str, Figure]
]
