from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, TypeVar

import torch
from artifact_experiment.base.callbacks.tracking import (
    ArrayCallbackHandler,
    ArrayCollectionCallbackHandler,
    PlotCallbackHandler,
    PlotCollectionCallbackHandler,
    ScoreCallbackHandler,
    ScoreCollectionCallbackHandler,
    TrackingCallbackHandler,
)
from matplotlib.figure import Figure
from numpy import ndarray
from tqdm import tqdm

from artifact_torch.base.components.callbacks.periodic import (
    PeriodicCacheCallback,
    PeriodicCallbackResources,
)
from artifact_torch.base.data.data_loader import DataLoader
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import (
    ModelInput,
    ModelOutput,
)

CacheDataT = TypeVar("CacheDataT")
BatchResultT = TypeVar("BatchResultT")
ModelInputT = TypeVar("ModelInputT", bound=ModelInput)
ModelOutputT = TypeVar("ModelOutputT", bound=ModelOutput)
DataLoaderCallbackT = TypeVar(
    "DataLoaderCallbackT",
    bound="DataLoaderCallback",
)


@dataclass
class DataLoaderCallbackResources(PeriodicCallbackResources, Generic[ModelInputT, ModelOutputT]):
    model: Model[ModelInputT, ModelOutputT]
    data_loader: DataLoader[ModelInputT]


class DataLoaderCallback(
    PeriodicCacheCallback[DataLoaderCallbackResources[ModelInputT, ModelOutputT], CacheDataT],
    Generic[ModelInputT, ModelOutputT, CacheDataT, BatchResultT],
):
    _verbose = True
    _progressbar_message = "Processing Data Loader"

    def __init__(self, execution_interval: int):
        key = self._get_key()
        super().__init__(key=key, period=execution_interval)
        self._ls_batch_results = []

    @classmethod
    @abstractmethod
    def _get_key(cls) -> str: ...

    @staticmethod
    @abstractmethod
    def _aggregate_batch_results(
        ls_batch_results: List[BatchResultT],
    ) -> CacheDataT: ...

    @staticmethod
    @abstractmethod
    def _compute_on_batch(
        model_input: ModelInputT,
        model_output: ModelOutputT,
    ) -> BatchResultT: ...

    def _compute(
        self,
        resources: DataLoaderCallbackResources[ModelInputT, ModelOutputT],
    ) -> CacheDataT:
        self._process_data_loader(model=resources.model, data_loader=resources.data_loader)
        result = self.finalize()
        return result

    def finalize(self) -> CacheDataT:
        result = self._aggregate_batch_results(ls_batch_results=self._ls_batch_results)
        self._ls_batch_results.clear()
        self._cache[self._key] = result
        return result

    def _process_data_loader(
        self,
        model: Model[ModelInputT, ModelOutputT],
        data_loader: DataLoader[ModelInputT],
    ):
        data_loader.device = model.device
        model.eval()
        with torch.no_grad():
            for model_input in tqdm(
                data_loader,
                desc=self._progressbar_message,
                disable=not self._verbose,
                leave=False,
            ):
                model_output = model(model_input)
                self.process_batch(model_input=model_input, model_output=model_output)

    def process_batch(
        self,
        model_input: ModelInputT,
        model_output: ModelOutputT,
    ):
        batch_result = self._compute_on_batch(model_input=model_input, model_output=model_output)
        self._ls_batch_results.append(batch_result)


DataLoaderScoreCallback = DataLoaderCallback[ModelInputT, ModelOutputT, float, float]
DataLoaderArrayCallback = DataLoaderCallback[ModelInputT, ModelOutputT, ndarray, ndarray]
DataLoaderPlotCallback = DataLoaderCallback[ModelInputT, ModelOutputT, Figure, float]
DataLoaderScoreCollectionCallback = DataLoaderCallback[
    ModelInputT, ModelOutputT, Dict[str, float], float
]
DataLoaderArrayCollectionCallback = DataLoaderCallback[
    ModelInputT, ModelOutputT, Dict[str, ndarray], ndarray
]
DataLoaderPlotCollectionCallback = DataLoaderCallback[
    ModelInputT, ModelOutputT, Dict[str, Figure], float
]


class DataLoaderCallbackHandler(
    TrackingCallbackHandler[
        DataLoaderCallback[ModelInputT, ModelOutputT, CacheDataT, Any],
        DataLoaderCallbackResources[ModelInputT, ModelOutputT],
        CacheDataT,
    ],
    Generic[
        ModelInputT,
        ModelOutputT,
        CacheDataT,
    ],
):
    _verbose = True
    _progressbar_message = "Processing Data Loader"

    def _execute(
        self,
        resources: DataLoaderCallbackResources[ModelInputT, ModelOutputT],
    ):
        if len(self._ls_callbacks) > 0:
            self._execute_parallel(resources=resources)
            self._finalize()

    def _execute_parallel(
        self,
        resources: DataLoaderCallbackResources[ModelInputT, ModelOutputT],
    ):
        resources.model.eval()
        with torch.no_grad():
            for model_input in tqdm(
                resources.data_loader,
                desc=self._progressbar_message,
                disable=not self._verbose,
                leave=False,
            ):
                model_output = resources.model(model_input)
                for callback in self._ls_callbacks:
                    callback.process_batch(model_input=model_input, model_output=model_output)

    def _finalize(self):
        for callback in self._ls_callbacks:
            callback.finalize()
        self.update_cache()

    def _execute_sequential(
        self,
        resources: DataLoaderCallbackResources[ModelInputT, ModelOutputT],
    ):
        super().execute(resources=resources)


class DataLoaderScoreHandler(
    DataLoaderCallbackHandler[
        ModelInputT,
        ModelOutputT,
        float,
    ],
    ScoreCallbackHandler[
        DataLoaderScoreCallback[ModelInputT, ModelOutputT],
        DataLoaderCallbackResources[ModelInputT, ModelOutputT],
    ],
    Generic[ModelInputT, ModelOutputT],
):
    pass


class DataLoaderArrayHandler(
    DataLoaderCallbackHandler[
        ModelInputT,
        ModelOutputT,
        ndarray,
    ],
    ArrayCallbackHandler[
        DataLoaderArrayCallback[ModelInputT, ModelOutputT],
        DataLoaderCallbackResources[ModelInputT, ModelOutputT],
    ],
    Generic[ModelInputT, ModelOutputT],
):
    pass


class DataLoaderPlotHandler(
    DataLoaderCallbackHandler[
        ModelInputT,
        ModelOutputT,
        Figure,
    ],
    PlotCallbackHandler[
        DataLoaderPlotCallback[ModelInputT, ModelOutputT],
        DataLoaderCallbackResources[ModelInputT, ModelOutputT],
    ],
    Generic[ModelInputT, ModelOutputT],
):
    pass


class DataLoaderScoreCollectionHandler(
    DataLoaderCallbackHandler[
        ModelInputT,
        ModelOutputT,
        Dict[str, float],
    ],
    ScoreCollectionCallbackHandler[
        DataLoaderScoreCollectionCallback[ModelInputT, ModelOutputT],
        DataLoaderCallbackResources[ModelInputT, ModelOutputT],
    ],
    Generic[ModelInputT, ModelOutputT],
):
    pass


class DataLoaderArrayCollectionHandler(
    DataLoaderCallbackHandler[
        ModelInputT,
        ModelOutputT,
        Dict[str, ndarray],
    ],
    ArrayCollectionCallbackHandler[
        DataLoaderArrayCollectionCallback[ModelInputT, ModelOutputT],
        DataLoaderCallbackResources[ModelInputT, ModelOutputT],
    ],
    Generic[ModelInputT, ModelOutputT],
):
    pass


class DataLoaderPlotCollectionHandler(
    DataLoaderCallbackHandler[
        ModelInputT,
        ModelOutputT,
        Dict[str, Figure],
    ],
    PlotCollectionCallbackHandler[
        DataLoaderPlotCollectionCallback[ModelInputT, ModelOutputT],
        DataLoaderCallbackResources[ModelInputT, ModelOutputT],
    ],
    Generic[ModelInputT, ModelOutputT],
):
    pass
