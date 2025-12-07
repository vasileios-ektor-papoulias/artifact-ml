from typing import Dict

import pytest
from artifact_core._libs.artifacts.binary_classification.threshold_variation.plotter import (
    ThresholdVariationCurvePlotter,
    ThresholdVariationCurveType,
)
from artifact_core._libs.resource_specs.binary_classification.protocol import (
    BinaryClassSpecProtocol,
)
from artifact_core._libs.resources.binary_classification.class_store import BinaryClassStore
from artifact_core._libs.resources.binary_classification.classification_results import (
    BinaryClassificationResults,
)
from artifact_core.binary_classification._artifacts.plot_collections.threshold_variation import (
    ThresholdVariationCurves,
    ThresholdVariationCurvesHyperparams,
)
from artifact_core.binary_classification._resources import BinaryClassificationArtifactResources
from matplotlib.figure import Figure
from pytest_mock import MockerFixture


@pytest.fixture
def hyperparams() -> ThresholdVariationCurvesHyperparams:
    return ThresholdVariationCurvesHyperparams(
        curve_types=[ThresholdVariationCurveType.ROC, ThresholdVariationCurveType.PR]
    )


@pytest.mark.unit
def test_compute(
    mocker: MockerFixture,
    resource_spec: BinaryClassSpecProtocol,
    true_class_store: BinaryClassStore,
    classification_results: BinaryClassificationResults,
    hyperparams: ThresholdVariationCurvesHyperparams,
):
    fake_plots: Dict[ThresholdVariationCurveType, Figure] = {
        ThresholdVariationCurveType.ROC: Figure(),
        ThresholdVariationCurveType.PR: Figure(),
    }
    patch_plot = mocker.patch.object(
        target=ThresholdVariationCurvePlotter,
        attribute="plot_multiple",
        return_value=fake_plots,
    )
    artifact = ThresholdVariationCurves(resource_spec=resource_spec, hyperparams=hyperparams)
    resources = BinaryClassificationArtifactResources.from_stores(
        true_class_store=true_class_store,
        classification_results=classification_results,
    )
    result = artifact.compute(resources=resources)
    patch_plot.assert_called_once_with(
        curve_types=hyperparams.curve_types,
        true=true_class_store.id_to_is_positive,
        probs=classification_results.id_to_prob_pos,
    )
    expected = {curve_type.value: plot for curve_type, plot in fake_plots.items()}
    assert result == expected
