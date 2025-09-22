from artifact_core.base.registry import ArtifactType


class BinaryClassificationPlotCollectionType(ArtifactType):
    CONFUSION_MATRIX_PLOTS = "confusion_matrix_plots"
    THRESHOLD_VARIATION_CURVES = "threshold_variation_curves"
