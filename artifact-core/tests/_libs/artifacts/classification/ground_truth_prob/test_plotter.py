from typing import Dict

import numpy as np
import pytest
from artifact_core._libs.artifacts.classification.ground_truth_prob.calculator import (  # noqa: E501
    GroundTruthProbCalculator,
)
from artifact_core._libs.artifacts.classification.ground_truth_prob.plotter import (  # noqa: E501
    GroundTruthProbPDFPlotter,
)
from artifact_core._libs.resources.classification.class_store import (  # noqa: E501
    ClassStore,
)
from artifact_core._libs.resources.classification.classification_results import (  # noqa: E501
    ClassificationResults,
)
from artifact_core._libs.tools.plotters.pdf_plotter import PDFPlotter
from artifact_core._utils.collections.entity_store import IdentifierType
from matplotlib.figure import Figure
from pytest_mock import MockerFixture


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_true_class, id_to_logits",
    [
        (
            {0: "A", 1: "B", 2: "C"},
            {
                0: np.array([2.0, 0.5, 0.1]),
                1: np.array([0.1, 2.5, 0.2]),
                2: np.array([0.2, 0.1, 3.0]),
            },
        ),
        (
            {0: "A"},
            {0: np.array([1.0, 0.5, 0.5])},
        ),
    ],
)
def test_plot(
    mocker: MockerFixture,
    set_agg_backend,
    close_all_figs_after_test,
    true_class_store: ClassStore,
    classification_results: ClassificationResults,
    id_to_true_class: Dict[IdentifierType, str],
    id_to_logits: Dict[IdentifierType, np.ndarray],
):
    for identifier, true_class in id_to_true_class.items():
        true_class_store.set_class(identifier=identifier, class_name=true_class)  # noqa: E501
        classification_results.set_single(
            identifier=identifier,
            predicted_class=true_class,
            logits=id_to_logits[identifier],
        )
    spy_calculator = mocker.spy(  # noqa: E501
        obj=GroundTruthProbCalculator, name="compute_id_to_prob_ground_truth"
    )
    spy_plotter = mocker.spy(obj=PDFPlotter, name="plot_pdf")
    result = GroundTruthProbPDFPlotter.plot(
        classification_results=classification_results,
        true_class_store=true_class_store,
    )
    assert isinstance(result, Figure)
    assert result.get_axes()
    spy_calculator.assert_called_once()
    spy_plotter.assert_called_once()
    assert spy_plotter.call_args.kwargs["feature_name"] == "P(y=ground_truth)"


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_true_class, id_to_logits",
    [
        (
            {0: "A", 1: "B", 2: "C"},
            {
                0: np.array([2.0, 0.5, 0.1]),
                1: np.array([0.1, 2.5, 0.2]),
                2: np.array([0.2, 0.1, 3.0]),
            },
        ),
    ],
)
def test_plot_title(
    set_agg_backend,
    close_all_figs_after_test,
    true_class_store: ClassStore,
    classification_results: ClassificationResults,
    id_to_true_class: Dict[IdentifierType, str],
    id_to_logits: Dict[IdentifierType, np.ndarray],
):
    for identifier, true_class in id_to_true_class.items():
        true_class_store.set_class(identifier=identifier, class_name=true_class)  # noqa: E501
        classification_results.set_single(
            identifier=identifier,
            predicted_class=true_class,
            logits=id_to_logits[identifier],
        )
    result = GroundTruthProbPDFPlotter.plot(
        classification_results=classification_results,
        true_class_store=true_class_store,
    )
    title_texts = [t.get_text() for t in result.texts]
    assert "PDF: P(y=ground_truth)" in title_texts
