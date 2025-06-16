from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TrackingScoreInfo:
    score_key: str
    latest_score: float
    best_score: float
    best_epoch: int


class TrainingProgressLogger:
    @staticmethod
    def get_progress_update(
        n_epochs_elapsed: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        tracking_score_info: Optional[TrackingScoreInfo] = None,
    ) -> str:
        desc = f"(Epoch {n_epochs_elapsed})"
        if train_loss is not None:
            desc += f"---> Train Loss: {train_loss:.4f}"
        if val_loss is not None:
            desc += f" | Val Loss: {val_loss:.4f}"
        if tracking_score_info is not None:
            if tracking_score_info.latest_score is not None and not np.isnan(
                tracking_score_info.latest_score
            ):
                score_recap = TrainingProgressLogger._get_score_recap(
                    score_key=tracking_score_info.score_key,
                    best_score=tracking_score_info.best_score,
                    best_epoch=tracking_score_info.best_epoch,
                )
                desc += (
                    f" |{tracking_score_info.score_key}: {tracking_score_info.latest_score:.4f}"
                    f" | {score_recap}"
                )
        return desc.strip()

    @staticmethod
    def _get_score_recap(score_key: str, best_score: float, best_epoch: int) -> str:
        return f"Best {score_key}: {best_score:.4f} @ (Epoch {best_epoch})"
