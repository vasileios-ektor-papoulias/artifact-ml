from artifact_core.base.registry import ArtifactType


class BinaryClassificationPlotCollectionType(ArtifactType):
    CONFUSION_MATRIX_PLOTS = "confusion_matrix_plots"
    THRESHOLD_VARIATION_CURVES = "threshold_variation_curves"
    SCORE_PDF_PLOTS = "score_pdf_plots"
