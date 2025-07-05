from typing import Dict, Optional

import numpy as np
from artifact_experiment.base.callbacks.base import CallbackResources
from artifact_experiment.base.callbacks.tracking import (
    ArrayCallback,
    ArrayCallbackHandler,
    ArrayCollectionCallback,
    ArrayCollectionCallbackHandler,
    PlotCallback,
    PlotCallbackHandler,
    PlotCollectionCallback,
    PlotCollectionCallbackHandler,
    ScoreCallback,
    ScoreCallbackHandler,
    ScoreCollectionCallback,
    ScoreCollectionCallbackHandler,
)
from artifact_experiment.base.tracking.client import TrackingClient
from matplotlib.figure import Figure


class DummyScoreCallback(ScoreCallback[CallbackResources]):
    def __init__(
        self, key: str, compute_value: float = 1.0, tracking_client: Optional[TrackingClient] = None
    ):
        super().__init__(key=key, tracking_client=tracking_client)
        self._compute_value = compute_value

    def _compute(self, resources: CallbackResources) -> float:
        _ = resources
        return self._compute_value


class DummyArrayCallback(ArrayCallback[CallbackResources]):
    def __init__(
        self,
        key: str,
        compute_value: Optional[np.ndarray] = None,
        tracking_client: Optional[TrackingClient] = None,
    ):
        super().__init__(key=key, tracking_client=tracking_client)
        self._compute_value = compute_value if compute_value is not None else np.array([1, 2, 3])

    def _compute(self, resources: CallbackResources) -> np.ndarray:
        _ = resources
        return self._compute_value


class DummyPlotCallback(PlotCallback[CallbackResources]):
    def __init__(
        self,
        key: str,
        compute_value: Optional[Figure] = None,
        tracking_client: Optional[TrackingClient] = None,
    ):
        super().__init__(key=key, tracking_client=tracking_client)
        self._compute_value = compute_value if compute_value is not None else Figure()

    def _compute(self, resources: CallbackResources) -> Figure:
        return self._compute_value


class DummyScoreCollectionCallback(ScoreCollectionCallback[CallbackResources]):
    def __init__(
        self,
        key: str,
        compute_value: Optional[Dict[str, float]] = None,
        tracking_client: Optional[TrackingClient] = None,
    ):
        super().__init__(key=key, tracking_client=tracking_client)
        self._compute_value = (
            compute_value if compute_value is not None else {"score1": 1.0, "score2": 2.0}
        )

    def _compute(self, resources: CallbackResources) -> Dict[str, float]:
        _ = resources
        return self._compute_value


class DummyArrayCollectionCallback(ArrayCollectionCallback[CallbackResources]):
    def __init__(
        self,
        key: str,
        compute_value: Optional[Dict[str, np.ndarray]] = None,
        tracking_client: Optional[TrackingClient] = None,
    ):
        super().__init__(key=key, tracking_client=tracking_client)
        self._compute_value = (
            compute_value
            if compute_value is not None
            else {"array1": np.array([1, 2]), "array2": np.array([3, 4])}
        )

    def _compute(self, resources: CallbackResources) -> Dict[str, np.ndarray]:
        _ = resources
        return self._compute_value


class DummyPlotCollectionCallback(PlotCollectionCallback[CallbackResources]):
    def __init__(
        self,
        key: str,
        compute_value: Optional[Dict[str, Figure]] = None,
        tracking_client: Optional[TrackingClient] = None,
    ):
        super().__init__(key=key, tracking_client=tracking_client)
        self._compute_value = (
            compute_value if compute_value is not None else {"plot1": Figure(), "plot2": Figure()}
        )

    def _compute(self, resources: CallbackResources) -> Dict[str, Figure]:
        _ = resources
        return self._compute_value


DummyScoreCallbackHandler = ScoreCallbackHandler[
    ScoreCallback[CallbackResources], CallbackResources
]
DummyArrayCallbackHandler = ArrayCallbackHandler[
    ArrayCallback[CallbackResources], CallbackResources
]
DummyPlotCallbackHandler = PlotCallbackHandler[PlotCallback[CallbackResources], CallbackResources]
DummyScoreCollectionCallbackHandler = ScoreCollectionCallbackHandler[
    ScoreCollectionCallback[CallbackResources], CallbackResources
]
DummyArrayCollectionCallbackHandler = ArrayCollectionCallbackHandler[
    ArrayCollectionCallback[CallbackResources], CallbackResources
]
DummyPlotCollectionCallbackHandler = PlotCollectionCallbackHandler[
    PlotCollectionCallback[CallbackResources], CallbackResources
]
