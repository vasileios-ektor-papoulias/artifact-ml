import os

from artifact_experiment.libs.tracking.clear_ml.adapter import ClearMLRunAdapter
from artifact_experiment.libs.tracking.clear_ml.loggers.base import ClearMLArtifactLogger


class ClearMLScoreLogger(ClearMLArtifactLogger[float]):
    _series_name: str = "score"

    def _append(self, artifact_path: str, artifact: float):
        iteration = self._get_score_iteration(
            run=self._run, path=artifact_path, series_name=self._series_name
        )
        self._run.log_score(
            value=artifact, title=artifact_path, series=self._series_name, iteration=iteration
        )

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("scores", artifact_name)

    @staticmethod
    def _get_score_iteration(run: ClearMLRunAdapter, path: str, series_name: str) -> int:
        score_store = run.get_exported_scores()
        try:
            score = score_store.get(path=path)
            score_series = score.get_series(series_name=series_name)
            iteration = score_series.n_entries
        except KeyError:
            iteration = 0
        return iteration
