from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

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
from artifact_experiment.base.data_split import (
    DataSplit,
    DataSplitSuffixAppender,
)
from artifact_experiment.base.tracking.client import TrackingClient
from matplotlib.figure import Figure
from numpy import ndarray
from torch.nn import Module
from torch.utils.hooks import RemovableHandle
from tqdm import tqdm

from artifact_torch.base.components.callbacks.periodic import (
    PeriodicCallbackResources,
    PeriodicTrackingCallback,
)
from artifact_torch.base.data.data_loader import DataLoader
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput, ModelOutput

ModelTCov = TypeVar("ModelTCov", bound=Model, covariant=True)
ModelInputTCov = TypeVar("ModelInputTCov", bound=ModelInput, covariant=True)


@dataclass
class DataLoaderHookCallbackResources(
    PeriodicCallbackResources, Generic[ModelTCov, ModelInputTCov]
):
    model: ModelTCov
    data_loader: DataLoader[ModelInputTCov]


ModelTContr = TypeVar("ModelTContr", bound=Model, contravariant=True)
ModelInputTContr = TypeVar("ModelInputTContr", bound=ModelInput, contravariant=True)
ModelOutputTContr = TypeVar("ModelOutputTContr", bound=ModelOutput, contravariant=True)
CacheDataT = TypeVar("CacheDataT")
HookResultT = TypeVar("HookResultT")


class DataLoaderHookCallback(
    PeriodicTrackingCallback[
        DataLoaderHookCallbackResources[ModelTContr, ModelInputTContr],
        CacheDataT,
    ],
    Generic[
        ModelTContr,
        ModelInputTContr,
        ModelOutputTContr,
        CacheDataT,
        HookResultT,
    ],
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
    def _hook(
        cls, module: Module, inputs: Tuple[Any, ...], output: Any
    ) -> Optional[HookResultT]: ...

    @classmethod
    @abstractmethod
    def _get_target_modules(cls, model: Module) -> List[Module]: ...

    @classmethod
    @abstractmethod
    def _aggregate_hook_results(
        cls,
        hook_results: Dict[str, List[HookResultT]],
    ) -> CacheDataT: ...

    @staticmethod
    @abstractmethod
    def _export(key: str, value: CacheDataT, tracking_client: TrackingClient): ...

    def attach_hooks(self, model: ModelTContr, n_epochs_elapsed: int):
        if self._should_trigger(step=n_epochs_elapsed):
            self._attach_hooks(model=model)

    def detach_hooks(self, n_epochs_elapsed: int):
        if self._should_trigger(step=n_epochs_elapsed):
            self._detach_hooks()

    def finalize(self, n_epochs_elapsed: int):
        if self._should_trigger(step=n_epochs_elapsed):
            self._finalize()

    def _compute(
        self,
        resources: DataLoaderHookCallbackResources[ModelTContr, ModelInputTContr],
    ) -> CacheDataT:
        self._process_data_loader(model=resources.model, data_loader=resources.data_loader)
        result = self._finalize()
        return result

    def _process_data_loader(
        self,
        model: ModelTContr,
        data_loader: DataLoader[ModelInputTContr],
    ):
        data_loader.device = model.device
        model.eval()
        self._attach_hooks(model=model)
        try:
            with torch.no_grad():
                for model_input in tqdm(
                    data_loader,
                    desc=self._progressbar_message,
                    disable=not self._verbose,
                    leave=False,
                ):
                    _ = model(model_input)
        finally:
            self._detach_hooks()

    def _attach_hooks(self, model: ModelTContr):
        sink = self._hook_results

        def _wrapped_hook(module: Module, inputs: Tuple[Any, ...], output: Any):
            ret = self._hook(module, inputs, output)
            if ret is not None:
                module_name = f"{module.__class__.__name__}"
                if module_name not in sink:
                    sink[module_name] = []
                sink[module_name].append(ret)

        for m in self._get_target_modules(model):
            self._handles.append(m.register_forward_hook(_wrapped_hook))

    def _detach_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _finalize(self) -> CacheDataT:
        result = self._aggregate_hook_results(hook_results=self._hook_results)
        self._hook_results.clear()
        self._cache[self._key] = result
        return result

    @classmethod
    def _get_key(cls, data_split: DataSplit) -> str:
        name = cls._get_name()
        key = DataSplitSuffixAppender.append_suffix(name=name, data_split=data_split)
        return key


class DataLoaderHookScoreCallback(
    ScoreExportMixin,
    DataLoaderHookCallback[
        ModelTContr, ModelInputTContr, ModelOutputTContr, float, Dict[str, torch.Tensor]
    ],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _hook(
        cls, module: Module, inputs: Tuple[Any, ...], output: Any
    ) -> Optional[Dict[str, torch.Tensor]]: ...

    @classmethod
    @abstractmethod
    def _get_target_modules(cls, model: Module) -> List[Module]: ...

    @classmethod
    @abstractmethod
    def _aggregate_hook_results(
        cls,
        hook_results: Dict[str, List[Dict[str, torch.Tensor]]],
    ) -> float: ...


class DataLoaderHookArrayCallback(
    ArrayExportMixin,
    DataLoaderHookCallback[
        ModelTContr, ModelInputTContr, ModelOutputTContr, ndarray, Dict[str, torch.Tensor]
    ],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _hook(
        cls, module: Module, inputs: Tuple[Any, ...], output: Any
    ) -> Optional[Dict[str, torch.Tensor]]: ...

    @classmethod
    @abstractmethod
    def _get_target_modules(cls, model: Module) -> List[Module]: ...

    @classmethod
    @abstractmethod
    def _aggregate_hook_results(
        cls,
        hook_results: Dict[str, List[Dict[str, torch.Tensor]]],
    ) -> ndarray: ...


class DataLoaderHookPlotCallback(
    PlotExportMixin,
    DataLoaderHookCallback[
        ModelTContr, ModelInputTContr, ModelOutputTContr, Figure, Dict[str, torch.Tensor]
    ],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _hook(
        cls, module: Module, inputs: Tuple[Any, ...], output: Any
    ) -> Optional[Dict[str, torch.Tensor]]: ...

    @classmethod
    @abstractmethod
    def _get_target_modules(cls, model: Module) -> List[Module]: ...

    @classmethod
    @abstractmethod
    def _aggregate_hook_results(
        cls,
        hook_results: Dict[str, List[Dict[str, torch.Tensor]]],
    ) -> Figure: ...


class DataLoaderHookScoreCollectionCallback(
    ScoreCollectionExportMixin,
    DataLoaderHookCallback[
        ModelTContr,
        ModelInputTContr,
        ModelOutputTContr,
        Dict[str, float],
        Dict[str, torch.Tensor],
    ],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _hook(
        cls, module: Module, inputs: Tuple[Any, ...], output: Any
    ) -> Optional[Dict[str, torch.Tensor]]: ...

    @classmethod
    @abstractmethod
    def _get_target_modules(cls, model: Module) -> List[Module]: ...

    @classmethod
    @abstractmethod
    def _aggregate_hook_results(
        cls,
        hook_results: Dict[str, List[Dict[str, torch.Tensor]]],
    ) -> Dict[str, float]: ...


class DataLoaderHookArrayCollectionCallback(
    ArrayCollectionExportMixin,
    DataLoaderHookCallback[
        ModelTContr,
        ModelInputTContr,
        ModelOutputTContr,
        Dict[str, ndarray],
        Dict[str, torch.Tensor],
    ],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _hook(
        cls, module: Module, inputs: Tuple[Any, ...], output: Any
    ) -> Optional[Dict[str, torch.Tensor]]: ...

    @classmethod
    @abstractmethod
    def _get_target_modules(cls, model: Module) -> List[Module]: ...

    @classmethod
    @abstractmethod
    def _aggregate_hook_results(
        cls,
        hook_results: Dict[str, List[Dict[str, torch.Tensor]]],
    ) -> Dict[str, ndarray]: ...


class DataLoaderHookPlotCollectionCallback(
    PlotCollectionExportMixin,
    DataLoaderHookCallback[
        ModelTContr,
        ModelInputTContr,
        ModelOutputTContr,
        Dict[str, Figure],
        Dict[str, torch.Tensor],
    ],
):
    @classmethod
    @abstractmethod
    def _get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def _hook(
        cls, module: Module, inputs: Tuple[Any, ...], output: Any
    ) -> Optional[Dict[str, torch.Tensor]]: ...

    @classmethod
    @abstractmethod
    def _get_target_modules(cls, model: Module) -> List[Module]: ...

    @classmethod
    @abstractmethod
    def _aggregate_hook_results(
        cls,
        hook_results: Dict[str, List[Dict[str, torch.Tensor]]],
    ) -> Dict[str, Figure]: ...


DataLoaderHookCallbackT = TypeVar(
    "DataLoaderHookCallbackT",
    bound="DataLoaderHookCallback",
)
ModelInputT = TypeVar("ModelInputT", bound=ModelInput)
ModelOutputT = TypeVar("ModelOutputT", bound=ModelOutput)
ModelT = TypeVar("ModelT", bound=Model)


class DataLoaderHookCallbackHandler(
    TrackingCallbackHandler[
        DataLoaderHookCallbackT,
        DataLoaderHookCallbackResources[ModelT, ModelInputT],
        CacheDataT,
    ],
    Generic[
        DataLoaderHookCallbackT,
        ModelT,
        ModelInputT,
        ModelOutputT,
        CacheDataT,
    ],
):
    _verbose = True
    _progressbar_message = "Processing Data Loader (Hooks)"

    @staticmethod
    @abstractmethod
    def _export(cache: Dict[str, CacheDataT], tracking_client: TrackingClient):
        pass

    def should_trigger(self, n_epochs_elapsed: int) -> bool:
        return self._should_trigger(n_epochs_elapsed=n_epochs_elapsed)

    def attach_hooks(self, model: ModelT, n_epochs_elapsed: int):
        self._attach_hooks(model=model, n_epochs_elapsed=n_epochs_elapsed)

    def detach_hooks(self, n_epochs_elapsed: int):
        self._detach_hooks(n_epochs_elapsed=n_epochs_elapsed)

    def finalize(self, n_epochs_elapsed: int):
        self._finalize(n_epochs_elapsed=n_epochs_elapsed)

    def export(self):
        if self._tracking_client is not None:
            self._export(
                cache=self.active_cache,
                tracking_client=self._tracking_client,
            )

    def _execute(
        self,
        resources: DataLoaderHookCallbackResources[ModelT, ModelInputT],
    ):
        if self._has_callbacks:
            self._execute_parallel(resources=resources)

    def _execute_sequential(
        self,
        resources: DataLoaderHookCallbackResources[ModelT, ModelInputT],
    ):
        super().execute(resources=resources)

    def _execute_parallel(
        self,
        resources: DataLoaderHookCallbackResources[ModelT, ModelInputT],
    ):
        if self._should_trigger(n_epochs_elapsed=resources.step):
            self._attach_hooks(model=resources.model, n_epochs_elapsed=resources.step)
            resources.model.eval()
            try:
                with torch.no_grad():
                    for model_input in tqdm(
                        resources.data_loader,
                        desc=self._progressbar_message,
                        disable=not self._verbose,
                        leave=False,
                    ):
                        _ = resources.model(model_input)
            finally:
                self._detach_hooks(n_epochs_elapsed=resources.step)
            self._finalize(n_epochs_elapsed=resources.step)

    def _should_trigger(self, n_epochs_elapsed: int) -> bool:
        ls_active_callbacks = [
            callback
            for callback in self._ls_callbacks
            if callback.should_trigger(step=n_epochs_elapsed)
        ]
        return len(ls_active_callbacks) > 0

    def _attach_hooks(self, model: ModelT, n_epochs_elapsed: int):
        for callback in self._ls_callbacks:
            callback.attach_hooks(model=model, n_epochs_elapsed=n_epochs_elapsed)

    def _detach_hooks(self, n_epochs_elapsed: int):
        for callback in self._ls_callbacks:
            callback.detach_hooks(n_epochs_elapsed=n_epochs_elapsed)

    def _finalize(self, n_epochs_elapsed: int):
        for callback in self._ls_callbacks:
            callback.finalize(n_epochs_elapsed=n_epochs_elapsed)
        self.update_cache()


class DataLoaderHookScoreHandler(
    ScoreHandlerExportMixin,
    DataLoaderHookCallbackHandler[
        DataLoaderHookScoreCallback[ModelT, ModelInputT, ModelOutputT],
        ModelT,
        ModelInputT,
        ModelOutputT,
        float,
    ],
    Generic[ModelT, ModelInputT, ModelOutputT],
):
    pass


class DataLoaderHookArrayHandler(
    ArrayHandlerExportMixin,
    DataLoaderHookCallbackHandler[
        DataLoaderHookArrayCallback[ModelT, ModelInputT, ModelOutputT],
        ModelT,
        ModelInputT,
        ModelOutputT,
        ndarray,
    ],
    Generic[ModelT, ModelInputT, ModelOutputT],
):
    pass


class DataLoaderHookPlotHandler(
    PlotHandlerExportMixin,
    DataLoaderHookCallbackHandler[
        DataLoaderHookPlotCallback[ModelT, ModelInputT, ModelOutputT],
        ModelT,
        ModelInputT,
        ModelOutputT,
        Figure,
    ],
    Generic[ModelT, ModelInputT, ModelOutputT],
):
    pass


class DataLoaderHookScoreCollectionHandler(
    ScoreCollectionHandlerExportMixin,
    DataLoaderHookCallbackHandler[
        DataLoaderHookScoreCollectionCallback[ModelT, ModelInputT, ModelOutputT],
        ModelT,
        ModelInputT,
        ModelOutputT,
        Dict[str, float],
    ],
    Generic[ModelT, ModelInputT, ModelOutputT],
):
    pass


class DataLoaderHookArrayCollectionHandler(
    ArrayCollectionHandlerExportMixin,
    DataLoaderHookCallbackHandler[
        DataLoaderHookArrayCollectionCallback[ModelT, ModelInputT, ModelOutputT],
        ModelT,
        ModelInputT,
        ModelOutputT,
        Dict[str, ndarray],
    ],
    Generic[ModelT, ModelInputT, ModelOutputT],
):
    pass


class DataLoaderHookPlotCollectionHandler(
    PlotCollectionHandlerExportMixin,
    DataLoaderHookCallbackHandler[
        DataLoaderHookPlotCollectionCallback[ModelT, ModelInputT, ModelOutputT],
        ModelT,
        ModelInputT,
        ModelOutputT,
        Dict[str, Figure],
    ],
    Generic[ModelT, ModelInputT, ModelOutputT],
):
    pass
