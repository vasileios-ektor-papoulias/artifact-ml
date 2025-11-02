from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Tuple, Type, TypeVar

import torch
import torch.nn as nn
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
from torch.utils.hooks import RemovableHandle

from artifact_torch.base.components.callbacks.periodic import (
    PeriodicCallbackResources,
    PeriodicTrackingCallback,
)
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput

ModelTCov = TypeVar("ModelTCov", bound=Model, covariant=True)
ModelInputTCov = TypeVar("ModelInputTCov", bound=ModelInput, covariant=True)


@dataclass
class ForwardHookCallbackResources(PeriodicCallbackResources, Generic[ModelTCov]):
    model: ModelTCov


ModelTContr = TypeVar("ModelTContr", bound=Model, contravariant=True)
CacheDataT = TypeVar("CacheDataT")
HookResultT = TypeVar("HookResultT")


class ForwardHookCallback(
    PeriodicTrackingCallback[ForwardHookCallbackResources[ModelTContr], CacheDataT],
    Generic[ModelTContr, CacheDataT, HookResultT],
):
    _verbose = True
    _progressbar_message = "Processing Data Loader (Hooks)"

    def __init__(self, period: int, data_split: DataSplit):
        key = self._get_key(data_split=data_split)
        super().__init__(key=key, period=period)
        self._hook_results: Dict[str, List[HookResultT]] = {}
        self._handles: List[RemovableHandle] = []

    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _get_layers(cls, model: ModelTContr) -> Sequence[nn.Module]: ...

    @classmethod
    @abstractmethod
    def _hook(
        cls, module: nn.Module, inputs: Tuple[Any, ...], output: Any
    ) -> Optional[HookResultT]: ...

    @classmethod
    @abstractmethod
    def _aggregate(cls, hook_results: Dict[str, List[HookResultT]]) -> CacheDataT: ...

    @staticmethod
    @abstractmethod
    def _export(key: str, value: CacheDataT, tracking_client: TrackingClient): ...

    def attach(self, resources: ForwardHookCallbackResources[ModelTContr]) -> bool:
        if self._should_trigger(step=resources.step):
            self._attach(model=resources.model)
            return True
        else:
            return False

    def _compute(self, resources: ForwardHookCallbackResources[ModelTContr]) -> CacheDataT:
        _ = resources
        result = self._finalize()
        self._detach()
        return result

    def _attach(self, model: ModelTContr):
        sink = self._hook_results

        def _wrapped_hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            hook_return_value = self._hook(module, inputs, output)
            if hook_return_value is not None:
                module_name = f"{module.__class__.__name__}"
                if module_name not in sink:
                    sink[module_name] = []
                sink[module_name].append(hook_return_value)

        for module in self._get_layers(model):
            self._handles.append(module.register_forward_hook(_wrapped_hook))

    def _finalize(self) -> CacheDataT:
        result = self._aggregate(hook_results=self._hook_results)
        self._hook_results.clear()
        self._cache[self._key] = result
        return result

    def _detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @classmethod
    def _get_key(cls, data_split: DataSplit) -> str:
        name = cls._get_name()
        key = DataSplitSuffixAppender.append_suffix(name=name, data_split=data_split)
        return key


class ForwardHookScore(
    ScoreExportMixin,
    ForwardHookCallback[ModelTContr, float, Dict[str, torch.Tensor]],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _get_layers(cls, model: ModelTContr) -> List[nn.Module]: ...

    @classmethod
    @abstractmethod
    def _hook(
        cls, module: nn.Module, inputs: Tuple[Any, ...], output: Any
    ) -> Optional[Dict[str, torch.Tensor]]: ...

    @classmethod
    @abstractmethod
    def _aggregate(cls, hook_results: Dict[str, List[Dict[str, torch.Tensor]]]) -> float: ...


class ForwardHookArray(
    ArrayExportMixin,
    ForwardHookCallback[ModelTContr, ndarray, Dict[str, torch.Tensor]],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _get_layers(cls, model: ModelTContr) -> List[nn.Module]: ...

    @classmethod
    @abstractmethod
    def _hook(
        cls, module: nn.Module, inputs: Tuple[Any, ...], output: Any
    ) -> Optional[Dict[str, torch.Tensor]]: ...

    @classmethod
    @abstractmethod
    def _aggregate(cls, hook_results: Dict[str, List[Dict[str, torch.Tensor]]]) -> ndarray: ...


class ForwardHookPlot(
    PlotExportMixin,
    ForwardHookCallback[ModelTContr, Figure, Dict[str, torch.Tensor]],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _get_layers(cls, model: ModelTContr) -> List[nn.Module]: ...

    @classmethod
    @abstractmethod
    def _hook(
        cls, module: nn.Module, inputs: Tuple[Any, ...], output: Any
    ) -> Optional[Dict[str, torch.Tensor]]: ...

    @classmethod
    @abstractmethod
    def _aggregate(cls, hook_results: Dict[str, List[Dict[str, torch.Tensor]]]) -> Figure: ...


class ForwardHookScoreCollection(
    ScoreCollectionExportMixin,
    ForwardHookCallback[
        ModelTContr,
        Dict[str, float],
        Dict[str, torch.Tensor],
    ],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _get_layers(cls, model: ModelTContr) -> List[nn.Module]: ...

    @classmethod
    @abstractmethod
    def _hook(
        cls, module: nn.Module, inputs: Tuple[Any, ...], output: Any
    ) -> Optional[Dict[str, torch.Tensor]]: ...

    @classmethod
    @abstractmethod
    def _aggregate(
        cls, hook_results: Dict[str, List[Dict[str, torch.Tensor]]]
    ) -> Dict[str, float]: ...


class ForwardHookArrayCollection(
    ArrayCollectionExportMixin,
    ForwardHookCallback[
        ModelTContr,
        Dict[str, ndarray],
        Dict[str, torch.Tensor],
    ],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _get_layers(cls, model: ModelTContr) -> List[nn.Module]: ...

    @classmethod
    @abstractmethod
    def _hook(
        cls, module: nn.Module, inputs: Tuple[Any, ...], output: Any
    ) -> Optional[Dict[str, torch.Tensor]]: ...

    @classmethod
    @abstractmethod
    def _aggregate(
        cls, hook_results: Dict[str, List[Dict[str, torch.Tensor]]]
    ) -> Dict[str, ndarray]: ...


class ForwardHookPlotCollection(
    PlotCollectionExportMixin,
    ForwardHookCallback[
        ModelTContr,
        Dict[str, Figure],
        Dict[str, torch.Tensor],
    ],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _get_layers(cls, model: ModelTContr) -> List[nn.Module]: ...

    @classmethod
    @abstractmethod
    def _hook(
        cls, module: nn.Module, inputs: Tuple[Any, ...], output: Any
    ) -> Optional[Dict[str, torch.Tensor]]: ...

    @classmethod
    @abstractmethod
    def _aggregate(
        cls, hook_results: Dict[str, List[Dict[str, torch.Tensor]]]
    ) -> Dict[str, Figure]: ...


ForwardHookCallbackT = TypeVar("ForwardHookCallbackT", bound="ForwardHookCallback")
ModelT = TypeVar("ModelT", bound=Model)


class ForwardHookCallbackHandler(
    TrackingCallbackHandler[
        ForwardHookCallbackT,
        ForwardHookCallbackResources[ModelT],
        CacheDataT,
    ],
    Generic[
        ForwardHookCallbackT,
        ModelT,
        CacheDataT,
    ],
):
    @staticmethod
    @abstractmethod
    def _export(cache: Dict[str, CacheDataT], tracking_client: TrackingClient): ...

    def attach(self, resources: ForwardHookCallbackResources[ModelTContr]) -> bool:
        any_attached = False
        for callback in self._ls_callbacks:
            any_attached |= callback.attach(resources=resources)
        return any_attached


class ForwardHookScoreHandler(
    ScoreHandlerExportMixin,
    ForwardHookCallbackHandler[ForwardHookScore[ModelT], ModelT, float],
    Generic[ModelT],
):
    pass


class ForwardHookArrayHandler(
    ArrayHandlerExportMixin,
    ForwardHookCallbackHandler[ForwardHookArray[ModelT], ModelT, ndarray],
    Generic[ModelT],
):
    pass


class ForwardHookPlotHandler(
    PlotHandlerExportMixin,
    ForwardHookCallbackHandler[ForwardHookPlot[ModelT], ModelT, Figure],
    Generic[ModelT],
):
    pass


class ForwardHookScoreCollectionHandler(
    ScoreCollectionHandlerExportMixin,
    ForwardHookCallbackHandler[ForwardHookScoreCollection[ModelT], ModelT, Dict[str, float]],
    Generic[ModelT],
):
    pass


class ForwardHookArrayCollectionHandler(
    ArrayCollectionHandlerExportMixin,
    ForwardHookCallbackHandler[ForwardHookArrayCollection[ModelT], ModelT, Dict[str, ndarray]],
    Generic[ModelT],
):
    pass


class ForwardHookPlotCollectionHandler(
    PlotCollectionHandlerExportMixin,
    ForwardHookCallbackHandler[ForwardHookPlotCollection[ModelT], ModelT, Dict[str, Figure]],
    Generic[ModelT],
):
    pass


ForwardHookHandlersT = TypeVar("ForwardHookHandlersT", bound="ForwardHookHandlers")


@dataclass
class ForwardHookHandlers(Generic[ModelT]):
    score_handler: ForwardHookScoreHandler[ModelT]
    array_handler: ForwardHookArrayHandler[ModelT]
    plot_handler: ForwardHookPlotHandler[ModelT]
    score_collection_handler: ForwardHookScoreCollectionHandler[ModelT]
    array_collection_handler: ForwardHookArrayCollectionHandler[ModelT]
    plot_collection_handler: ForwardHookPlotCollectionHandler[ModelT]

    @classmethod
    def build(
        cls: Type[ForwardHookHandlersT],
        ls_score_callbacks: List[ForwardHookScore[ModelT]],
        ls_array_callbacks: List[ForwardHookArray[ModelT]],
        ls_plot_callbacks: List[ForwardHookPlot[ModelT]],
        ls_score_collection_callbacks: List[ForwardHookScoreCollection[ModelT]],
        ls_array_collection_callbacks: List[ForwardHookArrayCollection[ModelT]],
        ls_plot_collection_callbacks: List[ForwardHookPlotCollection[ModelT]],
        tracking_client: Optional[TrackingClient] = None,
    ) -> ForwardHookHandlersT:
        handlers = cls(
            score_handler=ForwardHookScoreHandler[ModelT](
                ls_callbacks=ls_score_callbacks, tracking_client=tracking_client
            ),
            array_handler=ForwardHookArrayHandler[ModelT](
                ls_callbacks=ls_array_callbacks, tracking_client=tracking_client
            ),
            plot_handler=ForwardHookPlotHandler[ModelT](
                ls_callbacks=ls_plot_callbacks, tracking_client=tracking_client
            ),
            score_collection_handler=ForwardHookScoreCollectionHandler[ModelT](
                ls_callbacks=ls_score_collection_callbacks, tracking_client=tracking_client
            ),
            array_collection_handler=ForwardHookArrayCollectionHandler[ModelT](
                ls_callbacks=ls_array_collection_callbacks, tracking_client=tracking_client
            ),
            plot_collection_handler=ForwardHookPlotCollectionHandler[ModelT](
                ls_callbacks=ls_plot_collection_callbacks, tracking_client=tracking_client
            ),
        )
        return handlers
