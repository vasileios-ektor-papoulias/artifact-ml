import pytest
from artifact_core._base.core.hyperparams import NO_ARTIFACT_HYPERPARAMS
from artifact_core._libs.artifacts.binary_classification.score_distribution.plotter import (
    ScorePDFPlotter,
)
from artifact_core._libs.resource_specs.binary_classification.protocol import (
    BinaryClassSpecProtocol,
)
from artifact_core._libs.resources.binary_classification.class_store import BinaryClassStore
from artifact_core._libs.resources.binary_classification.classification_results import (
    BinaryClassificationResults,
)
from artifact_core.binary_classification._artifacts.plots.score_distribution import (
    ScoreDistributionPlot,
)
from artifact_core.binary_classification._resources import BinaryClassificationArtifactResources
from matplotlib.figure import Figure
from pytest_mock import MockerFixture


@pytest.mark.unit
def test_compute(
    mocker: MockerFixture,
    resource_spec: BinaryClassSpecProtocol,
    true_class_store: BinaryClassStore,
    classification_results: BinaryClassificationResults,
):
    fake_fig = Figure()
    patch_plot = mocker.patch.object(
        target=ScorePDFPlotter,
        attribute="plot_overlaid",
        return_value=fake_fig,
    )
    artifact = ScoreDistributionPlot(
        resource_spec=resource_spec, hyperparams=NO_ARTIFACT_HYPERPARAMS
    )
    resources = BinaryClassificationArtifactResources.from_stores(
        true_class_store=true_class_store,
        classification_results=classification_results,
    )
    result = artifact.compute(resources=resources)
    patch_plot.assert_called_once_with(
        id_to_is_pos=true_class_store.id_to_is_positive,
        id_to_prob_pos=classification_results.id_to_prob_pos,
    )
    assert result is fake_fig
