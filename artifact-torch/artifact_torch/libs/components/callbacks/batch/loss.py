from artifact_torch.base.components.callbacks.batch import (
    BatchCallback,
)
from artifact_torch.base.model.io import ModelInput, ModelOutput


class BatchLossCallback(BatchCallback[ModelInput, ModelOutput, float]):
    @classmethod
    def _get_key(cls):
        return "batch_loss"

    @staticmethod
    def _compute_on_batch(model_input: ModelInput, model_output: ModelOutput) -> float:
        _ = model_input
        t_loss = model_output.get("t_loss")
        assert t_loss is not None
        return t_loss.item()
