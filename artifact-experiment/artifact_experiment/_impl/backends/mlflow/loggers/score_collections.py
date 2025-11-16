import os
from typing import List

from artifact_core.typing import ScoreCollection
from mlflow.entities import Metric

from artifact_experiment._impl.backends.mlflow.adapter import MlflowRunAdapter
from artifact_experiment._impl.backends.mlflow.loggers.artifacts import MlflowArtifactLogger


class MlflowScoreCollectionLogger(MlflowArtifactLogger[ScoreCollection]):
    def _append(self, item_path: str, item: ScoreCollection):
        next_step = self._get_next_step(collection_path=item_path, ls_score_names=list(item.keys()))
        for score_name, score_value in item.items():
            backend_path = self._get_score_path(collection_path=item_path, score_name=score_name)
            self._run.log_score(backend_path=backend_path, value=score_value, step=next_step)

    def _get_next_step(self, collection_path: str, ls_score_names: List[str]) -> int:
        ls_score_paths = [
            self._get_score_path(collection_path=collection_path, score_name=score_name)
            for score_name in ls_score_names
        ]
        ls_collection_histories = self._get_ls_collection_histories(
            run=self._run, ls_paths=ls_score_paths
        )
        next_step = self._get_next_step_from_collection_histories(
            ls_collection_histories=ls_collection_histories
        )
        return next_step

    @staticmethod
    def _get_ls_collection_histories(
        run: MlflowRunAdapter, ls_paths: List[str]
    ) -> List[List[Metric]]:
        ls_collection_histories = []
        for path in ls_paths:
            ls_history = run.get_ls_score_history(backend_path=path)
            ls_collection_histories.append(ls_history)
        return ls_collection_histories

    @staticmethod
    def _get_next_step_from_collection_histories(
        ls_collection_histories: List[List[Metric]],
    ) -> int:
        return 1 + max([len(ls_history) for ls_history in ls_collection_histories])

    @staticmethod
    def _get_score_path(collection_path: str, score_name) -> str:
        score_path = os.path.join(collection_path, score_name)
        return score_path

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("score_collections", item_name)
