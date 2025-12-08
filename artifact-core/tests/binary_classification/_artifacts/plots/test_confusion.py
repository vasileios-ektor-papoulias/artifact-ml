import pytest
from artifact_core._libs.artifacts.binary_classification.confusion.calculator import (
    ConfusionMatrixNormalizationStrategy,
)
from artifact_core._libs.artifacts.binary_classification.confusion.plotter import (
    ConfusionMatrixPlotter,
)
from artifact_core._libs.resource_specs.binary_classification.protocol import (
    BinaryClassSpecProtocol,
)
from artifact_core._libs.resources.binary_classification.class_store import BinaryClassStore
from artifact_core._libs.resources.binary_classification.classification_results import (
    BinaryClassificationResults,
)
from artifact_core.binary_classification._artifacts.plots.confusion import (
    ConfusionMatrixPlot,
    ConfusionMatrixPlotHyperparams,
)
from artifact_core.binary_classification._resources import BinaryClassificationArtifactResources
from matplotlib.figure import Figure
from pytest_mock import MockerFixture


@pytest.fixture
def hyperparams() -> ConfusionMatrixPlotHyperparams:
    return ConfusionMatrixPlotHyperparams(normalization=ConfusionMatrixNormalizationStrategy.ALL)


@pytest.mark.unit
def test_compute(
    mocker: MockerFixture,
    resource_spec: BinaryClassSpecProtocol,
    true_class_store: BinaryClassStore,
    classification_results: BinaryClassificationResults,
    hyperparams: ConfusionMatrixPlotHyperparams,
):
    fake_fig = Figure()
    patch_plot = mocker.patch.object(
        target=ConfusionMatrixPlotter,
        attribute="plot",
        return_value=fake_fig,
    )
    artifact = ConfusionMatrixPlot(resource_spec=resource_spec, hyperparams=hyperparams)
    resources = BinaryClassificationArtifactResources.from_stores(
        true_class_store=true_class_store,
        classification_results=classification_results,
    )
    result = artifact.compute(resources=resources)
    patch_plot.assert_called_once_with(
        true=true_class_store.id_to_is_positive,
        predicted=classification_results.id_to_predicted_positive,
        normalization=hyperparams.normalization,
    )
    assert result is fake_fig
