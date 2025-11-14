from typing import Any, TypeVar

from artifact_core.typing import (
    Array,
    ArrayCollection,
    Plot,
    PlotCollection,
    Score,
    ScoreCollection,
)

from artifact_experiment._base.components.callbacks.tracking import (
    TrackingCallback,
    TrackingCallbackResources,
)
from artifact_experiment._base.components.handlers.cache import CacheCallbackHandler
from artifact_experiment._base.typing.tracking_data import TrackingData

TrackingCallbackTCov = TypeVar(
    "TrackingCallbackTCov", bound=TrackingCallback[Any, Any], covariant=True
)
TrackingCallbackResourcesTContr = TypeVar(
    "TrackingCallbackResourcesTContr", bound=TrackingCallbackResources, contravariant=True
)
CacheDataTCov = TypeVar("CacheDataTCov", bound=TrackingData, covariant=True)


TrackingCallbackHandler = CacheCallbackHandler[
    TrackingCallbackTCov, TrackingCallbackResourcesTContr, CacheDataTCov
]


TrackingScoreHandler = TrackingCallbackHandler[
    TrackingCallbackTCov, TrackingCallbackResourcesTContr, Score
]
TrackingArrayHandler = TrackingCallbackHandler[
    TrackingCallbackTCov, TrackingCallbackResourcesTContr, Array
]
TrackingPlotHandler = TrackingCallbackHandler[
    TrackingCallbackTCov, TrackingCallbackResourcesTContr, Plot
]
TrackingScoreCollectionHandler = TrackingCallbackHandler[
    TrackingCallbackTCov, TrackingCallbackResourcesTContr, ScoreCollection
]
TrackingArrayCollectionHandler = TrackingCallbackHandler[
    TrackingCallbackTCov, TrackingCallbackResourcesTContr, ArrayCollection
]
TrackingPlotCollectionHandler = TrackingCallbackHandler[
    TrackingCallbackTCov, TrackingCallbackResourcesTContr, PlotCollection
]
