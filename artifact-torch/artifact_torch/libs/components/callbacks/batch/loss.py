from artifact_torch.base.components.callbacks.batch import (
    BatchScoreCallback,
)
from artifact_torch.base.model.io import ModelInput, ModelOutput


class BatchLossCallback(BatchScoreCallback[ModelInput, ModelOutput]):
    _name = "BATCH_LOSS"

    @classmethod
    def _get_name(cls) -> str:
        return cls._name

    @classmethod
    def _compute_on_batch(
        cls,
        model_input: ModelInput,
        model_output: ModelOutput,
    ) -> float:
        _ = model_input
        t_loss = model_output.get("t_loss")
        assert t_loss is not None
        return t_loss.item()
