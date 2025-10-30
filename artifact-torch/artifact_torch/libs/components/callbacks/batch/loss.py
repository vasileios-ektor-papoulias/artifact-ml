from artifact_torch.base.components.callbacks.batch import (
    BatchScoreCallback,
)
from artifact_torch.base.model.io import ModelInput, ModelOutput


class BatchLossCallback(BatchScoreCallback[ModelInput, ModelOutput]):
    @classmethod
    def _get_key(cls):
        return "BATCH_LOSS"

    @staticmethod
    def _compute_on_batch(
        model_input: ModelInput,
        model_output: ModelOutput,
    ) -> float:
        _ = model_input
        t_loss = model_output.get("t_loss")
        assert t_loss is not None
        return t_loss.item()
