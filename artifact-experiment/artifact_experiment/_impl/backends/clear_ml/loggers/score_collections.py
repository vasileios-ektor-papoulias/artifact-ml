import os

from artifact_core.typing import ScoreCollection

from artifact_experiment._impl.backends.clear_ml.adapter import ClearMLRunAdapter
from artifact_experiment._impl.backends.clear_ml.loggers.artifacts import ClearMLArtifactLogger


class ClearMLScoreCollectionLogger(ClearMLArtifactLogger[ScoreCollection]):
    def _append(self, item_path: str, item: ScoreCollection):
        iteration = self._get_score_collection_iteration(run=self._run, path=item_path)
        for score_name, score_value in item.items():
            self._run.log_score(
                value=score_value, title=item_path, series=score_name, iteration=iteration
            )

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("score_collections", item_name)

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
