from artifact_torch.base.components.callbacks.checkpoint import CheckpointCallback


class TorchCheckpointCallback(CheckpointCallback):
    _checkpoint_name = "TORCH_CHECKPOINT"

    @classmethod
    def _get_checkpoint_name(cls) -> str:
        return cls._checkpoint_name
