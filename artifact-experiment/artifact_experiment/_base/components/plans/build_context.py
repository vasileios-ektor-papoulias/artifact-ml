from dataclasses import dataclass
from typing import Optional

from artifact_experiment._base.tracking.background.queue import TrackingQueue
from artifact_experiment._base.tracking.background.writer import (
    ArrayCollectionWriter,
    ArrayWriter,
    FileWriter,
    PlotCollectionWriter,
    PlotWriter,
    ScoreCollectionWriter,
    ScoreWriter,
)


@dataclass(frozen=True)
class PlanBuildContext:
    tracking_queue: Optional[TrackingQueue]

    @property
    def score_writer(self) -> Optional[ScoreWriter]:
        return self.tracking_queue.score_writer if self.tracking_queue is not None else None

    @property
    def array_writer(self) -> Optional[ArrayWriter]:
        return self.tracking_queue.array_writer if self.tracking_queue is not None else None

    @property
    def plot_writer(self) -> Optional[PlotWriter]:
        return self.tracking_queue.plot_writer if self.tracking_queue is not None else None

    @property
    def score_collection_writer(self) -> Optional[ScoreCollectionWriter]:
        return (
            self.tracking_queue.score_collection_writer if self.tracking_queue is not None else None
        )

    @property
    def array_collection_writer(self) -> Optional[ArrayCollectionWriter]:
        return (
            self.tracking_queue.array_collection_writer if self.tracking_queue is not None else None
        )

    @property
    def plot_collection_writer(self) -> Optional[PlotCollectionWriter]:
        return (
            self.tracking_queue.plot_collection_writer if self.tracking_queue is not None else None
        )

    @property
    def file_writer(self) -> Optional[FileWriter]:
        return self.tracking_queue.file_writer if self.tracking_queue is not None else None
