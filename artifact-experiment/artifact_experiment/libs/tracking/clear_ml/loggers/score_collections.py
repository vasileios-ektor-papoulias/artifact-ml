import os
from typing import Dict

from artifact_experiment.libs.tracking.clear_ml.adapter import ClearMLRunAdapter
from artifact_experiment.libs.tracking.clear_ml.loggers.base import ClearMLArtifactLogger


class ClearMLScoreCollectionLogger(ClearMLArtifactLogger[Dict[str, float]]):
    def _log(self, path: str, artifact: Dict[str, float]):
        iteration = self._get_score_collection_iteration(run=self._run, path=path)
        for score_name, score_value in artifact.items():
            self._run.log_score(
                value=score_value, title=path, series=score_name, iteration=iteration
            )

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("score_collections", artifact_name)

    @staticmethod
    def _get_score_collection_iteration(run: ClearMLRunAdapter, path: str) -> int:
        dict_all_scores = run.get_exported_scores()
        try:
            dict_score_history = dict_all_scores[path]
            iteration = max(
                len(series_history["x"]) for series_history in dict_score_history.values()
            )
        except KeyError:
            iteration = 0
        return iteration
