import os

from artifact_experiment.libs.tracking.clear_ml.adapter import ClearMLRunAdapter
from artifact_experiment.libs.tracking.clear_ml.loggers.base import ClearMLArtifactLogger


class ClearMLScoreLogger(ClearMLArtifactLogger[float]):
    _series_name: str = "score"

    def _log(self, path: str, artifact: float):
        iteration = self._get_score_iteration(run=self._run, path=path, series=self._series_name)
        self._run.log_score(
            value=artifact, title=path, series=self._series_name, iteration=iteration
        )

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("scores", artifact_name)

    @staticmethod
    def _get_score_iteration(run: ClearMLRunAdapter, path: str, series: str) -> int:
        dict_all_scores = run.get_exported_scores()
        try:
            dict_score_history = dict_all_scores[path][series]
            iteration = len(dict_score_history["x"])
        except KeyError:
            iteration = 0
        return iteration
