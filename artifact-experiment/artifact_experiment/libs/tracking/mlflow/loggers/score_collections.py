import os
from typing import Dict, List

from mlflow.entities import Metric

from artifact_experiment.libs.tracking.mlflow.adapter import MlflowRunAdapter
from artifact_experiment.libs.tracking.mlflow.loggers.base import MlflowArtifactLogger


class MlflowScoreCollectionLogger(MlflowArtifactLogger[Dict[str, float]]):
    def _log(self, path: str, artifact: Dict[str, float]):
        next_step = self._get_next_step(collection_path=path, ls_score_names=list(artifact.keys()))
        for score_name, score_value in artifact.items():
            backend_path = self._get_score_path(collection_path=path, score_name=score_name)
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
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("score_collections", artifact_name)
