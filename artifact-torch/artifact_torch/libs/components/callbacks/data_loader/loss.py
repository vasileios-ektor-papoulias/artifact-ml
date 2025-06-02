from typing import List

from artifact_torch.base.components.callbacks.data_loader import (
    DataLoaderScoreCallback,
)
from artifact_torch.base.model.io import ModelInput, ModelOutput


class LoaderLossCallback(DataLoaderScoreCallback[ModelInput, ModelOutput]):
    @staticmethod
    def _aggregate_batch_results(ls_batch_results: List[float]) -> float:
        return sum(ls_batch_results) / len(ls_batch_results) if ls_batch_results else float("nan")

    @staticmethod
    def _compute_on_batch(model_input: ModelInput, model_output: ModelOutput) -> float:
        _ = model_input
        t_loss = model_output.get("t_loss")
        assert t_loss is not None
        return t_loss.item()


class TrainLossCallback(LoaderLossCallback):
    @classmethod
    def _get_key(cls):
        return "train_loss"


class ValLossCallback(LoaderLossCallback):
    @classmethod
    def _get_key(cls):
        return "val_loss"
