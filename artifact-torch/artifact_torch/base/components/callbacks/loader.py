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
from artifact_experiment.base.data_split import DataSplit, DataSplitSuffixAppender
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
from artifact_torch.base.model.io import ModelInput, ModelOutput

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
CacheDataT = TypeVar("CacheDataT")
BatchResultT = TypeVar("BatchResultT")


class DataLoaderCallback(
    PeriodicTrackingCallback[
        DataLoaderCallbackResources[ModelInputTContr, ModelOutputTContr], CacheDataT
    ],
    Generic[ModelInputTContr, ModelOutputTContr, CacheDataT, BatchResultT],
):
    _verbose = True
    _progressbar_message = "Processing Data Loader"

    def __init__(self, period: int, data_split: DataSplit):
        key = self._get_key(data_split=data_split)
        super().__init__(key=key, period=period)
        self._ls_batch_results: List[BatchResultT] = []

    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _compute_on_batch(
        cls,
        model_input: ModelInputTContr,
        model_output: ModelOutputTContr,
    ) -> BatchResultT: ...

    @classmethod
    @abstractmethod
    def _aggregate_batch_results(
        cls,
        ls_batch_results: List[BatchResultT],
    ) -> CacheDataT: ...

    @staticmethod
    @abstractmethod
    def _export(key: str, value: CacheDataT, tracking_client: TrackingClient): ...

    def finalize(
        self,
        n_epochs_elapsed: int,
    ):
        if self._should_trigger(step=n_epochs_elapsed):
            self._finalize()

    def process_batch(
        self,
        model_input: ModelInputTContr,
        model_output: ModelOutputTContr,
        n_epochs_elapsed: int,
    ):
        if self._should_trigger(step=n_epochs_elapsed):
            self._process_batch(model_input=model_input, model_output=model_output)

    def _compute(
        self,
        resources: DataLoaderCallbackResources[ModelInputTContr, ModelOutputTContr],
    ) -> CacheDataT:
        self._process_data_loader(model=resources.model, data_loader=resources.data_loader)
        result = self._finalize()
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
                self._process_batch(model_input=model_input, model_output=model_output)

    def _process_batch(
        self,
        model_input: ModelInputTContr,
        model_output: ModelOutputTContr,
    ):
        batch_result = self._compute_on_batch(model_input=model_input, model_output=model_output)
        self._ls_batch_results.append(batch_result)

    def _finalize(self) -> CacheDataT:
        result = self._aggregate_batch_results(ls_batch_results=self._ls_batch_results)
        self._ls_batch_results.clear()
        self._cache[self._key] = result
        return result

    @classmethod
    def _get_key(cls, data_split: DataSplit) -> str:
        name = cls._get_name()
        key = DataSplitSuffixAppender.append_suffix(name=name, data_split=data_split)
        return key


class DataLoaderScoreCallback(
    ScoreExportMixin,
    DataLoaderCallback[ModelInputTContr, ModelOutputTContr, float, Dict[str, torch.Tensor]],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _compute_on_batch(
        cls,
        model_input: ModelInputTContr,
        model_output: ModelOutputTContr,
    ) -> Dict[str, torch.Tensor]: ...

    @classmethod
    @abstractmethod
    def _aggregate_batch_results(
        cls,
        ls_batch_results: List[Dict[str, torch.Tensor]],
    ) -> float: ...


class DataLoaderArrayCallback(
    ArrayExportMixin,
    DataLoaderCallback[ModelInputTContr, ModelOutputTContr, ndarray, Dict[str, torch.Tensor]],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _compute_on_batch(
        cls,
        model_input: ModelInputTContr,
        model_output: ModelOutputTContr,
    ) -> Dict[str, torch.Tensor]: ...

    @classmethod
    @abstractmethod
    def _aggregate_batch_results(
        cls,
        ls_batch_results: List[Dict[str, torch.Tensor]],
    ) -> ndarray: ...


class DataLoaderPlotCallback(
    PlotExportMixin,
    DataLoaderCallback[ModelInputTContr, ModelOutputTContr, Figure, Dict[str, torch.Tensor]],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _compute_on_batch(
        cls,
        model_input: ModelInputTContr,
        model_output: ModelOutputTContr,
    ) -> Dict[str, torch.Tensor]: ...

    @classmethod
    @abstractmethod
    def _aggregate_batch_results(
        cls,
        ls_batch_results: List[Dict[str, torch.Tensor]],
    ) -> Figure: ...


class DataLoaderScoreCollectionCallback(
    ScoreCollectionExportMixin,
    DataLoaderCallback[
        ModelInputTContr, ModelOutputTContr, Dict[str, float], Dict[str, torch.Tensor]
    ],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _compute_on_batch(
        cls,
        model_input: ModelInputTContr,
        model_output: ModelOutputTContr,
    ) -> Dict[str, torch.Tensor]: ...

    @classmethod
    @abstractmethod
    def _aggregate_batch_results(
        cls,
        ls_batch_results: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, float]: ...


class DataLoaderArrayCollectionCallback(
    ArrayCollectionExportMixin,
    DataLoaderCallback[
        ModelInputTContr, ModelOutputTContr, Dict[str, ndarray], Dict[str, torch.Tensor]
    ],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _compute_on_batch(
        cls,
        model_input: ModelInputTContr,
        model_output: ModelOutputTContr,
    ) -> Dict[str, torch.Tensor]: ...

    @classmethod
    @abstractmethod
    def _aggregate_batch_results(
        cls,
        ls_batch_results: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, ndarray]: ...


class DataLoaderPlotCollectionCallback(
    PlotCollectionExportMixin,
    DataLoaderCallback[
        ModelInputTContr, ModelOutputTContr, Dict[str, Figure], Dict[str, torch.Tensor]
    ],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _compute_on_batch(
        cls,
        model_input: ModelInputTContr,
        model_output: ModelOutputTContr,
    ) -> Dict[str, torch.Tensor]: ...

    @classmethod
    @abstractmethod
    def _aggregate_batch_results(
        cls,
        ls_batch_results: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, Figure]: ...


DataLoaderCallbackT = TypeVar(
    "DataLoaderCallbackT",
    bound="DataLoaderCallback",
)
ModelInputT = TypeVar("ModelInputT", bound=ModelInput)
ModelOutputT = TypeVar("ModelOutputT", bound=ModelOutput)


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

    def should_trigger(self, n_epochs_elapsed: int) -> bool:
        return self._should_trigger(n_epochs_elapsed=n_epochs_elapsed)

    def process_batch(
        self, model_input: ModelInputT, model_output: ModelOutputT, n_epochs_elapsed: int
    ):
        self._process_batch(
            model_input=model_input, model_output=model_output, n_epochs_elapsed=n_epochs_elapsed
        )

    def finalize(self, n_epochs_elapsed: int):
        self._finalize(n_epochs_elapsed=n_epochs_elapsed)

    def export(self):
        if self._tracking_client is not None:
            self._export(cache=self.active_cache, tracking_client=self._tracking_client)

    def _execute(
        self,
        resources: DataLoaderCallbackResources[ModelInputT, ModelOutputT],
    ):
        self._execute_parallel(resources=resources)

    def _execute_sequential(
        self,
        resources: DataLoaderCallbackResources[ModelInputT, ModelOutputT],
    ):
        super().execute(resources=resources)

    def _execute_parallel(
        self,
        resources: DataLoaderCallbackResources[ModelInputT, ModelOutputT],
    ):
        if self._should_trigger(n_epochs_elapsed=resources.step):
            resources.model.eval()
            with torch.no_grad():
                for model_input in tqdm(
                    resources.data_loader,
                    desc=self._progressbar_message,
                    disable=not self._verbose,
                    leave=False,
                ):
                    model_output = resources.model(model_input)
                    self._process_batch(
                        model_input=model_input,
                        model_output=model_output,
                        n_epochs_elapsed=resources.step,
                    )
            self._finalize(n_epochs_elapsed=resources.step)

    def _should_trigger(self, n_epochs_elapsed: int) -> bool:
        ls_active_callbacks = [
            callback
            for callback in self._ls_callbacks
            if callback.should_trigger(step=n_epochs_elapsed)
        ]
        return len(ls_active_callbacks) > 0

    def _process_batch(
        self, model_input: ModelInputT, model_output: ModelOutputT, n_epochs_elapsed: int
    ):
        for callback in self._ls_callbacks:
            callback.process_batch(
                model_input=model_input,
                model_output=model_output,
                n_epochs_elapsed=n_epochs_elapsed,
            )

    def _finalize(self, n_epochs_elapsed: int):
        for callback in self._ls_callbacks:
            callback.finalize(n_epochs_elapsed=n_epochs_elapsed)
        self.update_cache()


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
