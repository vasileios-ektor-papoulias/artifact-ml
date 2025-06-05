import os
from typing import Dict

from artifact_experiment.libs.tracking.clear_ml.adapter import ClearMLRunAdapter
from artifact_experiment.libs.tracking.clear_ml.loggers.base import ClearMLArtifactLogger


class ClearMLScoreCollectionLogger(ClearMLArtifactLogger[Dict[str, float]]):
    def _append(self, artifact_path: str, artifact: Dict[str, float]):
        iteration = self._get_score_collection_iteration(run=self._run, path=artifact_path)
        for score_name, score_value in artifact.items():
            self._run.log_score(
                value=score_value, title=artifact_path, series=score_name, iteration=iteration
            )

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("score_collections", artifact_name)

    @staticmethod
    def _get_score_collection_iteration(run: ClearMLRunAdapter, path: str) -> int:
        score_store = run.get_exported_scores()
        try:
            score = score_store.get(path=path)
            ls_series_iterations = [
                len(score_series) for score_series in score.dict_series.values()
            ]
            iteration = max(ls_series_iterations)
        except KeyError:
            iteration = 0
        return iteration
