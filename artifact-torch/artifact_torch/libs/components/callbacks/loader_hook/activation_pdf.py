from typing import Any, Dict, List, Optional, Tuple, TypeVar

import pandas as pd
import torch
from artifact_core.libs.utils.plotters.pdf_plotter import PDFConfig, PDFPlotter
from matplotlib.figure import Figure
from torch import Tensor
from torch.nn import Module

from artifact_torch.base.components.callbacks.loader_hook import DataLoaderHookPlotCallback
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput, ModelOutput
from artifact_torch.libs.utils.tensor_flattener import TensorFlattener

ModelInputT = TypeVar("ModelInputT", bound=ModelInput)
ModelOutputT = TypeVar("ModelOutputT", bound=ModelOutput)


class AllActivationsPDFCallback(
    DataLoaderHookPlotCallback[
        Model[ModelInputT, ModelOutputT],
        ModelInputT,
        ModelOutputT,
    ]
):
    _name = "ACTIVATIONS_PDF"
    _hook_result_key = "layer_activations"
    _plot_feature_name = "All Layers — Activations"
    _max_activation_samples = 1_000_000

    @classmethod
    def _get_name(cls) -> str:
        return cls._name

    @classmethod
    def _get_target_modules(cls, model: Module) -> List[Module]:
        return [m for m in model.modules() if not any(m.children())]

    @classmethod
    def _hook(
        cls, module: Module, inputs: Tuple[Any, ...], output: Any
    ) -> Optional[Dict[str, Tensor]]:
        _ = module
        _ = inputs
        ls_t_out: List[Tensor] = []
        for t in TensorFlattener.flatten_tensors(x=output):
            ls_t_out.append(t.detach().reshape(-1).cpu())
        if ls_t_out:
            t_layer_activations = torch.cat(ls_t_out, dim=0)
            return {cls._hook_result_key: t_layer_activations}

    @classmethod
    def _aggregate_hook_results(
        cls,
        hook_results: Dict[str, List[Dict[str, Tensor]]],
    ) -> Figure:
        collected = cls._collect_activations(hook_results)
        sr = cls._activations_to_series(collected)
        return cls._plot_pdf(sr)

    @classmethod
    def _collect_activations(
        cls,
        hook_results: Dict[str, List[Dict[str, Tensor]]],
    ) -> List[Tensor]:
        collected: List[Tensor] = []
        for ls_activations in hook_results.values():
            for activation in ls_activations:
                vals = activation.get(cls._hook_result_key, None)
                if vals is not None and vals.numel() > 0:
                    collected.append(vals)
        return collected

    @classmethod
    def _sample_activations(cls, t_activations: Tensor) -> Tensor:
        n_activations = t_activations.numel()
        if n_activations <= cls._max_activation_samples:
            return t_activations
        idx = torch.randperm(n_activations)[: cls._max_activation_samples]
        return t_activations.index_select(0, idx)

    @classmethod
    def _activations_to_series(cls, collected: List[Tensor]) -> pd.Series:
        if not collected:
            return pd.Series(dtype=float)
        t_all = torch.cat(collected, dim=0)
        t_all = cls._sample_activations(t_all)
        all_vals = t_all.numpy()
        return pd.Series(all_vals, dtype=float)

    @classmethod
    def _plot_pdf(cls, sr: pd.Series) -> Figure:
        return PDFPlotter.plot_pdf(
            sr_data=sr,
            feature_name=cls._plot_feature_name,
            config=PDFConfig(),
        )
