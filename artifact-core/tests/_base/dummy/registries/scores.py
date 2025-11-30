from typing import Any, Dict

from artifact_core._base.typing.artifact_result import Score

from tests._base.dummy.registries.base import DummyArtifactRegistry
from tests._base.dummy.types.scores import DummyScoreType


class DummyScoreRegistry(DummyArtifactRegistry[DummyScoreType, Score]):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "DUMMY_SCORE_ARTIFACT": {"adjust_scale": True},
            "CUSTOM_SCORE_ARTIFACT": {"result": 0},
        }


class AlternativeDummyScoreRegistry(DummyArtifactRegistry[DummyScoreType, Score]):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class MissingParamDummyScoreRegistry(DummyArtifactRegistry[DummyScoreType, Score]):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "DUMMY_SCORE_ARTIFACT": {},
        }


class InvalidParamDummyScoreRegistry(DummyArtifactRegistry[DummyScoreType, Score]):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "DUMMY_SCORE_ARTIFACT": {"invalid": 0},
        }
