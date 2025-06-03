from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Generic, List, TypeVar

import torch
from artifact_experiment.base.callbacks.tracking import (
    ArrayCollectionExportMixin,
    ArrayCollectionHandlerExportMixin,
    ArrayExportMixin,
    ArrayHandlerExportMixin,
    PlotCollectionExportMixin,
    PlotCollectionHandlerExportMixin,
    PlotExportMixin,
    PlotHandlerExportMixin,
    ScoreCollectionExportMixin,
    ScoreCollectionHandlerExportMixin,
    ScoreExportMixin,
    ScoreHandlerExportMixin,
    TrackingCallbackHandler,
)
from artifact_experiment.base.tracking.client import TrackingClient
from matplotlib.figure import Figure
from numpy import ndarray
from tqdm import tqdm

from artifact_torch.base.components.callbacks.periodic import (
    PeriodicCallbackResources,
    PeriodicTrackingCallback,
)
from artifact_torch.base.data.data_loader import DataLoader
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import (
    ModelInput,
    ModelOutput,
)

CacheDataT = TypeVar("CacheDataT")
BatchResultT = TypeVar("BatchResultT")
ModelInputTCov = TypeVar("ModelInputTCov", bound=ModelInput, covariant=True)
ModelOutputTCov = TypeVar("ModelOutputTCov", bound=ModelOutput, covariant=True)


@dataclass
class DataLoaderCallbackResources(
    PeriodicCallbackResources, Generic[ModelInputTCov, ModelOutputTCov]
):
    model: Model[ModelInputTCov, ModelOutputTCov]
    data_loader: DataLoader[ModelInputTCov]


ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)


class DataLoaderCallback(
    PeriodicTrackingCallback[
        DataLoaderCallbackResources[ModelInputTContr, ModelOutputTContr], CacheDataT
    ],
    Generic[ModelInputTContr, ModelOutputTContr, CacheDataT, BatchResultT],
):
    _verbose = True
    _progressbar_message = "Processing Data Loader"

    def __init__(self, period: int):
        key = self._get_key()
        super().__init__(key=key, period=period)
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
        model_input: ModelInputTContr,
        model_output: ModelOutputTContr,
    ) -> BatchResultT: ...

    @staticmethod
    @abstractmethod
    def _export(key: str, value: CacheDataT, tracking_client: TrackingClient): ...

    def finalize(self) -> CacheDataT:
        result = self._aggregate_batch_results(ls_batch_results=self._ls_batch_results)
        self._ls_batch_results.clear()
        self._cache[self._key] = result
        return result

    def _compute(
        self,
        resources: DataLoaderCallbackResources[ModelInputTContr, ModelOutputTContr],
    ) -> CacheDataT:
        self._process_data_loader(model=resources.model, data_loader=resources.data_loader)
        result = self.finalize()
        return result

    def _process_data_loader(
        self,
        model: Model[ModelInputTContr, ModelOutputTContr],
        data_loader: DataLoader[ModelInputTContr],
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
        model_input: ModelInputTContr,
        model_output: ModelOutputTContr,
    ):
        batch_result = self._compute_on_batch(model_input=model_input, model_output=model_output)
        self._ls_batch_results.append(batch_result)


class DataLoaderScoreCallback(
    ScoreExportMixin, DataLoaderCallback[ModelInputTContr, ModelOutputTContr, float, float]
):
    @classmethod
    @abstractmethod
    def _get_key(cls) -> str: ...

    @staticmethod
    @abstractmethod
    def _aggregate_batch_results(
        ls_batch_results: List[float],
    ) -> float: ...

    @staticmethod
    @abstractmethod
    def _compute_on_batch(
        model_input: ModelInputTContr,
        model_output: ModelOutputTContr,
    ) -> float: ...


class DataLoaderArrayCallback(
    ArrayExportMixin, DataLoaderCallback[ModelInputTContr, ModelOutputTContr, ndarray, ndarray]
):
    @classmethod
    @abstractmethod
    def _get_key(cls) -> str: ...

    @staticmethod
    @abstractmethod
    def _aggregate_batch_results(
        ls_batch_results: List[ndarray],
    ) -> ndarray: ...

    @staticmethod
    @abstractmethod
    def _compute_on_batch(
        model_input: ModelInputTContr,
        model_output: ModelOutputTContr,
    ) -> ndarray: ...


class DataLoaderPlotCallback(
    PlotExportMixin, DataLoaderCallback[ModelInputTContr, ModelOutputTContr, Figure, ndarray]
):
    @classmethod
    @abstractmethod
    def _get_key(cls) -> str: ...

    @staticmethod
    @abstractmethod
    def _aggregate_batch_results(
        ls_batch_results: List[ndarray],
    ) -> Figure: ...

    @staticmethod
    @abstractmethod
    def _compute_on_batch(
        model_input: ModelInputTContr,
        model_output: ModelOutputTContr,
    ) -> ndarray: ...


class DataLoaderScoreCollectionCallback(
    ScoreCollectionExportMixin,
    DataLoaderCallback[ModelInputTContr, ModelOutputTContr, Dict[str, float], Dict[str, float]],
):
    @classmethod
    @abstractmethod
    def _get_key(cls) -> str: ...

    @staticmethod
    @abstractmethod
    def _aggregate_batch_results(
        ls_batch_results: List[Dict[str, float]],
    ) -> Dict[str, float]: ...

    @staticmethod
    @abstractmethod
    def _compute_on_batch(
        model_input: ModelInputTContr,
        model_output: ModelOutputTContr,
    ) -> Dict[str, float]: ...


class DataLoaderArrayCollectionCallback(
    ArrayCollectionExportMixin,
    DataLoaderCallback[ModelInputTContr, ModelOutputTContr, Dict[str, ndarray], Dict[str, ndarray]],
):
    @classmethod
    @abstractmethod
    def _get_key(cls) -> str: ...

    @staticmethod
    @abstractmethod
    def _aggregate_batch_results(
        ls_batch_results: List[Dict[str, ndarray]],
    ) -> Dict[str, ndarray]: ...

    @staticmethod
    @abstractmethod
    def _compute_on_batch(
        model_input: ModelInputTContr,
        model_output: ModelOutputTContr,
    ) -> Dict[str, ndarray]: ...


class DataLoaderPlotCollectionCallback(
    PlotCollectionExportMixin,
    DataLoaderCallback[ModelInputTContr, ModelOutputTContr, Dict[str, Figure], Dict[str, ndarray]],
):
    @classmethod
    @abstractmethod
    def _get_key(cls) -> str: ...

    @staticmethod
    @abstractmethod
    def _aggregate_batch_results(
        ls_batch_results: List[Dict[str, ndarray]],
    ) -> Dict[str, Figure]: ...

    @staticmethod
    @abstractmethod
    def _compute_on_batch(
        model_input: ModelInputTContr,
        model_output: ModelOutputTContr,
    ) -> Dict[str, ndarray]: ...


ModelInputT = TypeVar("ModelInputT", bound=ModelInput)
ModelOutputT = TypeVar("ModelOutputT", bound=ModelOutput)
DataLoaderCallbackT = TypeVar(
    "DataLoaderCallbackT",
    bound="DataLoaderCallback",
)


class DataLoaderCallbackHandler(
    TrackingCallbackHandler[
        DataLoaderCallbackT,
        DataLoaderCallbackResources[ModelInputT, ModelOutputT],
        CacheDataT,
    ],
    Generic[
        DataLoaderCallbackT,
        ModelInputT,
        ModelOutputT,
        CacheDataT,
    ],
):
    _verbose = True
    _progressbar_message = "Processing Data Loader"

    @staticmethod
    @abstractmethod
    def _export(cache: Dict[str, CacheDataT], tracking_client: TrackingClient):
        pass

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
    ScoreHandlerExportMixin,
    DataLoaderCallbackHandler[
        DataLoaderScoreCallback[ModelInputT, ModelOutputT],
        ModelInputT,
        ModelOutputT,
        float,
    ],
    Generic[ModelInputT, ModelOutputT],
):
    pass


class DataLoaderArrayHandler(
    ArrayHandlerExportMixin,
    DataLoaderCallbackHandler[
        DataLoaderArrayCallback[ModelInputT, ModelOutputT],
        ModelInputT,
        ModelOutputT,
        ndarray,
    ],
    Generic[ModelInputT, ModelOutputT],
):
    pass


class DataLoaderPlotHandler(
    PlotHandlerExportMixin,
    DataLoaderCallbackHandler[
        DataLoaderPlotCallback[ModelInputT, ModelOutputT],
        ModelInputT,
        ModelOutputT,
        Figure,
    ],
    Generic[ModelInputT, ModelOutputT],
):
    pass


class DataLoaderScoreCollectionHandler(
    ScoreCollectionHandlerExportMixin,
    DataLoaderCallbackHandler[
        DataLoaderScoreCollectionCallback[ModelInputT, ModelOutputT],
        ModelInputT,
        ModelOutputT,
        Dict[str, float],
    ],
    Generic[ModelInputT, ModelOutputT],
):
    pass


class DataLoaderArrayCollectionHandler(
    ArrayCollectionHandlerExportMixin,
    DataLoaderCallbackHandler[
        DataLoaderArrayCollectionCallback[ModelInputT, ModelOutputT],
        ModelInputT,
        ModelOutputT,
        Dict[str, ndarray],
    ],
    Generic[ModelInputT, ModelOutputT],
):
    pass


class DataLoaderPlotCollectionHandler(
    PlotCollectionHandlerExportMixin,
    DataLoaderCallbackHandler[
        DataLoaderPlotCollectionCallback[ModelInputT, ModelOutputT],
        ModelInputT,
        ModelOutputT,
        Dict[str, Figure],
    ],
    Generic[ModelInputT, ModelOutputT],
):
    pass
