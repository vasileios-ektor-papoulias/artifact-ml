from typing import Dict

from artifact_experiment.libs.tracking.clear_ml.loggers.base import ClearMLArtifactLogger


class ClearMLScoreCollectionLogger(ClearMLArtifactLogger[Dict[str, float]]):
    def _log(self, path: str, artifact: Dict[str, float]):
        for score_name, score_value in artifact.items():
            self._run.log_score(
                value=score_value, title=path, series=score_name, iteration=self._iteration
            )
            self._iteration += 1

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"score_collections/{artifact_name}"
