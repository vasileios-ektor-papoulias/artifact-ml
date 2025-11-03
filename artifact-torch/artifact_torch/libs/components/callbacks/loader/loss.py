from typing import Dict, List

import torch

from artifact_torch.base.components.callbacks.model_io import ModelIOScoreCallback
from artifact_torch.base.model.io import ModelInput, ModelOutput


class LoaderLossCallback(ModelIOScoreCallback[ModelInput, ModelOutput]):
    _name = "LOSS"

    @classmethod
    def _get_name(cls):
        return cls._name

    @classmethod
    def _compute_on_batch(
        cls,
        model_input: ModelInput,
        model_output: ModelOutput,
    ) -> Dict[str, torch.Tensor]:
        _ = model_input
        t_loss = model_output.get("t_loss")
        assert t_loss is not None, "Loss tensor not provided"
        return {"t_loss": t_loss}

    @classmethod
    def _aggregate_batch_results(cls, ls_batch_results: List[Dict[str, torch.Tensor]]) -> float:
        ls_batch_losses = [result["t_loss"].item() for result in ls_batch_results]
        return sum(ls_batch_losses) / len(ls_batch_losses) if ls_batch_losses else float("nan")
