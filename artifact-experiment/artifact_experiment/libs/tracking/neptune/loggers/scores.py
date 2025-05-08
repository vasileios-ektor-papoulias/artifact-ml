from artifact_experiment.libs.tracking.neptune.loggers.base import NeptuneArtifactLogger


class NeptuneScoreLogger(NeptuneArtifactLogger[float]):
    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"scores/{artifact_name}"
