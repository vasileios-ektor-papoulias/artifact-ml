from typing import Dict

import pytest
from artifact_core._libs.artifacts.binary_classification.score_distribution.partitioner import (
    BinarySampleSplit,
)
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
from artifact_core.binary_classification._artifacts.plot_collections.score_distribution import (
    ScoreDistributionPlots,
    ScoreDistributionPlotsHyperparams,
)
from artifact_core.binary_classification._resources import BinaryClassificationArtifactResources
from matplotlib.figure import Figure
from pytest_mock import MockerFixture


@pytest.fixture
def hyperparams() -> ScoreDistributionPlotsHyperparams:
    return ScoreDistributionPlotsHyperparams(
        split_types=[BinarySampleSplit.POSITIVE, BinarySampleSplit.NEGATIVE]
    )


@pytest.mark.unit
def test_compute(
    mocker: MockerFixture,
    resource_spec: BinaryClassSpecProtocol,
    true_class_store: BinaryClassStore,
    classification_results: BinaryClassificationResults,
    hyperparams: ScoreDistributionPlotsHyperparams,
):
    fake_plots: Dict[BinarySampleSplit, Figure] = {
        BinarySampleSplit.POSITIVE: Figure(),
        BinarySampleSplit.NEGATIVE: Figure(),
    }
    patch_plot = mocker.patch.object(
        target=ScorePDFPlotter,
        attribute="plot_multiple",
        return_value=fake_plots,
    )
    artifact = ScoreDistributionPlots(resource_spec=resource_spec, hyperparams=hyperparams)
    resources = BinaryClassificationArtifactResources.from_stores(
        true_class_store=true_class_store,
        classification_results=classification_results,
    )
    result = artifact.compute(resources=resources)
    patch_plot.assert_called_once_with(
        id_to_is_pos=true_class_store.id_to_is_positive,
        id_to_prob_pos=classification_results.id_to_prob_pos,
        splits=hyperparams.split_types,
    )
    expected = {split_type.value: plot for split_type, plot in fake_plots.items()}
    assert result == expected
