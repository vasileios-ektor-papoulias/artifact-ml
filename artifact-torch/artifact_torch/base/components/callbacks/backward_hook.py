from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Generic, List, Optional, Tuple, TypeVar

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
from artifact_experiment.base.entities.data_split import DataSplit, DataSplitSuffixAppender
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
class BackwardHookCallbackResources(PeriodicCallbackResources, Generic[ModelTCov]):
    model: ModelTCov


ModelTContr = TypeVar("ModelTContr", bound=Model, contravariant=True)
CacheDataT = TypeVar("CacheDataT")
HookResultT = TypeVar("HookResultT")


class BackwardHookCallback(
    PeriodicTrackingCallback[BackwardHookCallbackResources[ModelTContr], CacheDataT],
    Generic[ModelTContr, CacheDataT, HookResultT],
):
    _verbose = True
    _progressbar_message = "Processing Data Loader (Backward Hooks)"

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
    def _get_layers(cls, model: ModelTContr) -> List[nn.Module]: ...

    @classmethod
    @abstractmethod
    def _hook(
        cls,
        module: nn.Module,
        grad_input: Tuple[Optional[torch.Tensor], ...],
        grad_output: Tuple[Optional[torch.Tensor], ...],
    ) -> Optional[HookResultT]: ...

    @classmethod
    @abstractmethod
    def _aggregate(cls, hook_results: Dict[str, List[HookResultT]]) -> CacheDataT: ...

    @staticmethod
    @abstractmethod
    def _export(key: str, value: CacheDataT, tracking_client: TrackingClient): ...

    def attach(self, resources: BackwardHookCallbackResources[ModelTContr]):
        if self._should_trigger(step=resources.step):
            self._attach(model=resources.model)

    def _compute(self, resources: BackwardHookCallbackResources[ModelTContr]) -> CacheDataT:
        _ = resources
        result = self._finalize()
        self._detach()
        return result

    def _attach(self, model: ModelTContr):
        sink = self._hook_results

        def _wrapped_hook(
            module: nn.Module,
            grad_input: Tuple[Optional[torch.Tensor], ...],
            grad_output: Tuple[Optional[torch.Tensor], ...],
        ):
            hook_return_value = self._hook(module, grad_input, grad_output)
            if hook_return_value is not None:
                module_name = f"{module.__class__.__name__}"
                if module_name not in sink:
                    sink[module_name] = []
                sink[module_name].append(hook_return_value)

        for module in self._get_layers(model):
            handle = module.register_full_backward_hook(_wrapped_hook)  # type: ignore[reportArgumentType]
            self._handles.append(handle)

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


class BackwardHookScore(
    ScoreExportMixin,
    BackwardHookCallback[ModelTContr, float, Dict[str, torch.Tensor]],
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
        cls,
        module: nn.Module,
        grad_input: Tuple[Optional[torch.Tensor], ...],
        grad_output: Tuple[Optional[torch.Tensor], ...],
    ) -> Optional[Dict[str, torch.Tensor]]: ...

    @classmethod
    @abstractmethod
    def _aggregate(cls, hook_results: Dict[str, List[Dict[str, torch.Tensor]]]) -> float: ...


class BackwardHookArray(
    ArrayExportMixin,
    BackwardHookCallback[ModelTContr, ndarray, Dict[str, torch.Tensor]],
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
        cls,
        module: nn.Module,
        grad_input: Tuple[Optional[torch.Tensor], ...],
        grad_output: Tuple[Optional[torch.Tensor], ...],
    ) -> Optional[Dict[str, torch.Tensor]]: ...

    @classmethod
    @abstractmethod
    def _aggregate(cls, hook_results: Dict[str, List[Dict[str, torch.Tensor]]]) -> ndarray: ...


class BackwardHookPlot(
    PlotExportMixin,
    BackwardHookCallback[ModelTContr, Figure, Dict[str, torch.Tensor]],
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
        cls,
        module: nn.Module,
        grad_input: Tuple[Optional[torch.Tensor], ...],
        grad_output: Tuple[Optional[torch.Tensor], ...],
    ) -> Optional[Dict[str, torch.Tensor]]: ...

    @classmethod
    @abstractmethod
    def _aggregate(cls, hook_results: Dict[str, List[Dict[str, torch.Tensor]]]) -> Figure: ...


class BackwardHookScoreCollection(
    ScoreCollectionExportMixin,
    BackwardHookCallback[
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
        cls,
        module: nn.Module,
        grad_input: Tuple[Optional[torch.Tensor], ...],
        grad_output: Tuple[Optional[torch.Tensor], ...],
    ) -> Optional[Dict[str, torch.Tensor]]: ...

    @classmethod
    @abstractmethod
    def _aggregate(
        cls, hook_results: Dict[str, List[Dict[str, torch.Tensor]]]
    ) -> Dict[str, float]: ...


class BackwardHookArrayCollection(
    ArrayCollectionExportMixin,
    BackwardHookCallback[
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
        cls,
        module: nn.Module,
        grad_input: Tuple[Optional[torch.Tensor], ...],
        grad_output: Tuple[Optional[torch.Tensor], ...],
    ) -> Optional[Dict[str, torch.Tensor]]: ...

    @classmethod
    @abstractmethod
    def _aggregate(
        cls, hook_results: Dict[str, List[Dict[str, torch.Tensor]]]
    ) -> Dict[str, ndarray]: ...


class BackwardHookPlotCollection(
    PlotCollectionExportMixin,
    BackwardHookCallback[
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
        cls,
        module: nn.Module,
        grad_input: Tuple[Optional[torch.Tensor], ...],
        grad_output: Tuple[Optional[torch.Tensor], ...],
    ) -> Optional[Dict[str, torch.Tensor]]: ...

    @classmethod
    @abstractmethod
    def _aggregate(
        cls, hook_results: Dict[str, List[Dict[str, torch.Tensor]]]
    ) -> Dict[str, Figure]: ...


BackwardHookCallbackT = TypeVar("BackwardHookCallbackT", bound="BackwardHookCallback")
ModelT = TypeVar("ModelT", bound=Model)


class BackwardHookCallbackHandler(
    TrackingCallbackHandler[
        BackwardHookCallbackT,
        BackwardHookCallbackResources[ModelT],
        CacheDataT,
    ],
    Generic[
        BackwardHookCallbackT,
        ModelT,
        CacheDataT,
    ],
):
    @staticmethod
    @abstractmethod
    def _export(cache: Dict[str, CacheDataT], tracking_client: TrackingClient): ...

    def attach(self, resources: BackwardHookCallbackResources[ModelTContr]):
        for callback in self._ls_callbacks:
            callback.attach(resources=resources)


class BackwardHookScoreHandler(
    ScoreHandlerExportMixin,
    BackwardHookCallbackHandler[BackwardHookScore[ModelT], ModelT, float],
    Generic[ModelT],
):
    pass


class BackwardHookArrayHandler(
    ArrayHandlerExportMixin,
    BackwardHookCallbackHandler[BackwardHookArray[ModelT], ModelT, ndarray],
    Generic[ModelT],
):
    pass


class BackwardHookPlotHandler(
    PlotHandlerExportMixin,
    BackwardHookCallbackHandler[BackwardHookPlot[ModelT], ModelT, Figure],
    Generic[ModelT],
):
    pass


class BackwardHookScoreCollectionHandler(
    ScoreCollectionHandlerExportMixin,
    BackwardHookCallbackHandler[BackwardHookScoreCollection[ModelT], ModelT, Dict[str, float]],
    Generic[ModelT],
):
    pass


class BackwardHookArrayCollectionHandler(
    ArrayCollectionHandlerExportMixin,
    BackwardHookCallbackHandler[BackwardHookArrayCollection[ModelT], ModelT, Dict[str, ndarray]],
    Generic[ModelT],
):
    pass


class BackwardHookPlotCollectionHandler(
    PlotCollectionHandlerExportMixin,
    BackwardHookCallbackHandler[BackwardHookPlotCollection[ModelT], ModelT, Dict[str, Figure]],
    Generic[ModelT],
):
    pass
