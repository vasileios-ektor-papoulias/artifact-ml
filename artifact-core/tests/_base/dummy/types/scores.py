from artifact_core._base.orchestration.registry import ArtifactType


class DummyScoreType(ArtifactType):
    DUMMY_SCORE_ARTIFACT = "dummy_score_artifact"
    NO_HYPERPARAMS_ARTIFACT = "no_hyperparams_artifact"
    IN_ALTERNATIVE_REGISTRY = "in_alternative_registry"
    NOT_REGISTERED = "not_registered"
