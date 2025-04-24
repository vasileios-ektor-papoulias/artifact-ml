from artifact_core.base.registry import ArtifactType


class TableComparisonArrayCollectionType(ArtifactType):
    MEANS = "means"
    STDS = "stds"
    VARIANCES = "variances"
    MEDIANS = "medians"
    FIRST_QUARTILES = "first_quartiles"
    THIRD_QUARTILES = "third_quartiles"
    MINIMA = "minima"
    MAXIMA = "maxima"
