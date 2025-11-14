from artifact_torch.base.components.early_stopping.stopper import EarlyStopper, StopperUpdateData


class EpochBoundStopper(EarlyStopper[StopperUpdateData]):
    def __init__(self, max_n_epochs: int):
        super().__init__(max_n_epochs=max_n_epochs)

    def stopping_condition_met(self) -> bool:
        max_epochs_exceeded = self._max_epochs_exceeded()
        return max_epochs_exceeded
