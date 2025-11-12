from typing import Dict, Optional

import numpy as np
from artifact_experiment.base.components.callbacks.base import CallbackResources
from artifact_experiment.base.components.callbacks.tracking import (
    ArrayCallback,
    ArrayCollectionCallback,
    PlotCallback,
    PlotCollectionCallback,
    ScoreCallback,
    ScoreCollectionCallback,
)
from artifact_experiment.base.tracking.backend.client import TrackingClient


class DummyScoreCallback(ScoreCallback[CallbackResources]):
    DEFAULT_VALUE = 1.0

    def __init__(
        self,
        key: str,
        compute_value: Optional[float] = None,
        tracking_client: Optional[TrackingClient] = None,
    ):
        super().__init__(key=key, tracking_client=tracking_client)
        self._compute_value = compute_value if compute_value is not None else self.DEFAULT_VALUE

    def _compute(self, resources: CallbackResources) -> float:
        _ = resources
        return self._compute_value


class DummyArrayCallback(ArrayCallback[CallbackResources]):
    DEFAULT_VALUE = np.array([1, 2, 3])

    def __init__(
        self,
        key: str,
        compute_value: Optional[Array] = None,
        tracking_client: Optional[TrackingClient] = None,
    ):
        super().__init__(key=key, tracking_client=tracking_client)
        self._compute_value = compute_value if compute_value is not None else self.DEFAULT_VALUE

    def _compute(self, resources: CallbackResources) -> Array:
        _ = resources
        return self._compute_value


class DummyPlotCallback(PlotCallback[CallbackResources]):
    DEFAULT_VALUE = Figure()

    def __init__(
        self,
        key: str,
        compute_value: Optional[Figure] = None,
        tracking_client: Optional[TrackingClient] = None,
    ):
        super().__init__(key=key, tracking_client=tracking_client)
        self._compute_value = compute_value if compute_value is not None else self.DEFAULT_VALUE

    def _compute(self, resources: CallbackResources) -> Figure:
        _ = resources
        return self._compute_value


class DummyScoreCollectionCallback(ScoreCollectionCallback[CallbackResources]):
    DEFAULT_VALUE = {"score1": 1.0, "score2": 2.0}

    def __init__(
        self,
        key: str,
        compute_value: Optional[Dict[str, float]] = None,
        tracking_client: Optional[TrackingClient] = None,
    ):
        super().__init__(key=key, tracking_client=tracking_client)
        self._compute_value = compute_value if compute_value is not None else self.DEFAULT_VALUE

    def _compute(self, resources: CallbackResources) -> Dict[str, float]:
        _ = resources
        return self._compute_value


class DummyArrayCollectionCallback(ArrayCollectionCallback[CallbackResources]):
    DEFAULT_VALUE = {"array1": np.array([1, 2]), "array2": np.array([3, 4])}

    def __init__(
        self,
        key: str,
        compute_value: Optional[Dict[str, Array]] = None,
        tracking_client: Optional[TrackingClient] = None,
    ):
        super().__init__(key=key, tracking_client=tracking_client)
        self._compute_value = compute_value if compute_value is not None else self.DEFAULT_VALUE

    def _compute(self, resources: CallbackResources) -> Dict[str, Array]:
        _ = resources
        return self._compute_value


class DummyPlotCollectionCallback(PlotCollectionCallback[CallbackResources]):
    DEFAULT_VALUE = {"plot1": Figure(), "plot2": Figure()}

    def __init__(
        self,
        key: str,
        compute_value: Optional[Dict[str, Figure]] = None,
        tracking_client: Optional[TrackingClient] = None,
    ):
        super().__init__(key=key, tracking_client=tracking_client)
        self._compute_value = compute_value if compute_value is not None else self.DEFAULT_VALUE

    def _compute(self, resources: CallbackResources) -> Dict[str, Figure]:
        _ = resources
        return self._compute_value
