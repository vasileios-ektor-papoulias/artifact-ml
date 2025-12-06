from typing import Type

import pytest
from artifact_core._libs.resource_specs.binary_classification.protocol import (
    BinaryClassSpecProtocol,
)
from artifact_core.binary_classification._artifacts.base import BinaryClassificationPlotCollection
from artifact_core.binary_classification._artifacts.plot_collections.confusion import (
    ConfusionMatrixPlotCollection,
)
from artifact_core.binary_classification._artifacts.plot_collections.score_distribution import (
    ScoreDistributionPlots,
)
from artifact_core.binary_classification._artifacts.plot_collections.threshold_variation import (
    ThresholdVariationCurves,
)
from artifact_core.binary_classification._registries.plot_collections import (
    BinaryClassificationPlotCollectionRegistry,
)
from artifact_core.binary_classification._types.plot_collections import (
    BinaryClassificationPlotCollectionType,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, artifact_class",
    [
        (
            BinaryClassificationPlotCollectionType.CONFUSION_MATRIX_PLOTS,
            ConfusionMatrixPlotCollection,
        ),
        (
            BinaryClassificationPlotCollectionType.THRESHOLD_VARIATION_CURVES,
            ThresholdVariationCurves,
        ),
        (BinaryClassificationPlotCollectionType.SCORE_PDF_PLOTS, ScoreDistributionPlots),
    ],
)
def test_get(
    resource_spec: BinaryClassSpecProtocol,
    artifact_type: BinaryClassificationPlotCollectionType,
    artifact_class: Type[BinaryClassificationPlotCollection],
):
    artifact = BinaryClassificationPlotCollectionRegistry.get(
        artifact_type=artifact_type, resource_spec=resource_spec
    )
    assert isinstance(artifact, artifact_class)
    assert artifact.resource_spec == resource_spec
