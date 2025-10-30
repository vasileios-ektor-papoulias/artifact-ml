from artifact_torch.base.components.callbacks.batch import (
    BatchScoreCallback,
)
from artifact_torch.base.model.base import Model
from artifact_torch.base.model.io import ModelInput, ModelOutput


class BatchLossCallback(BatchScoreCallback[ModelInput, ModelOutput]):
    @classmethod
    def _get_key(cls):
        return "batch_loss"

    @staticmethod
    def _compute_on_batch(
        model_input: ModelInput, model_output: ModelOutput, model: Model
    ) -> float:
        _ = model_input
        _ = model
        t_loss = model_output.get("t_loss")
        assert t_loss is not None
        return t_loss.item()
