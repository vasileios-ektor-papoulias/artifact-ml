from artifact_torch.base.components.early_stopping.single_score import (
    SingleScoreStopper,
)


class ScoreMinimizationStopper(SingleScoreStopper):
    def stopping_condition_met(self) -> bool:
        max_epochs_exceeded = self._max_epochs_exceeded()
        queue_is_full = self._queue_is_full()
        return max_epochs_exceeded or (queue_is_full and self.min_score == self.earliest_score)


class ScoreMaximizationStopper(SingleScoreStopper):
    def stopping_condition_met(self) -> bool:
        max_epochs_exceeded = self._max_epochs_exceeded()
        queue_is_full = self._queue_is_full()
        return max_epochs_exceeded or (queue_is_full and self.max_score == self.earliest_score)
