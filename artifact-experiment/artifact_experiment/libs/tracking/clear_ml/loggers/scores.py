import os

from artifact_experiment.libs.tracking.clear_ml.adapter import ClearMLRunAdapter
from artifact_experiment.libs.tracking.clear_ml.loggers.base import ClearMLArtifactLogger
from artifact_experiment.libs.tracking.clear_ml.readers.scores import ClearMLScoreReader


class ClearMLScoreLogger(ClearMLArtifactLogger[float]):
    _series_name: str = "score"

    def _append(self, artifact_path: str, artifact: float):
        iteration = self._get_score_iteration(
            run=self._run, path=artifact_path, series=self._series_name
        )
        self._run.log_score(
            value=artifact, title=artifact_path, series=self._series_name, iteration=iteration
        )

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("scores", artifact_name)

    @staticmethod
    def _get_score_iteration(run: ClearMLRunAdapter, path: str, series: str) -> int:
        try:
            dict_score_history = ClearMLScoreReader.get_score_history(run=run, score_path=path)
            dict_score_series = ClearMLScoreReader.get_series_from_score_history(
                score_history=dict_score_history, series_name=series
            )
            ls_xvalues = ClearMLScoreReader.get_xvalues_from_score_series(
                score_series=dict_score_series
            )
            iteration = len(ls_xvalues)
        except KeyError:
            iteration = 0
        return iteration
