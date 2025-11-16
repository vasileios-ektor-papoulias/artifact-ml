import os

from artifact_core.typing import Score

from artifact_experiment._impl.backends.clear_ml.adapter import ClearMLRunAdapter
from artifact_experiment._impl.backends.clear_ml.loggers.artifacts import ClearMLArtifactLogger


class ClearMLScoreLogger(ClearMLArtifactLogger[Score]):
    _series_name: str = "score"

    def _append(self, item_path: str, item: Score):
        iteration = self._get_score_iteration(
            run=self._run, path=item_path, series_name=self._series_name
        )
        self._run.log_score(
            value=item, title=item_path, series=self._series_name, iteration=iteration
        )

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("scores", item_name)

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
