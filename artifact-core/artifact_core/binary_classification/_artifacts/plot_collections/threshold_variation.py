from dataclasses import dataclass
from typing import Dict, Sequence, Type, TypeVar, Union

from matplotlib.figure import Figure

from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._libs.artifacts.binary_classification.threshold_variation.plotter import (
    ThresholdVariationCurvePlotter,
    ThresholdVariationCurveType,
    ThresholdVariationCurveTypeLiteral,
)
from artifact_core._libs.resources.binary_classification.class_store import BinaryClassStore
from artifact_core._libs.resources.binary_classification.classification_results import (
    BinaryClassificationResults,
)
from artifact_core.binary_classification._artifacts.base import BinaryClassificationPlotCollection
from artifact_core.binary_classification._registries.plot_collections import (
    BinaryClassificationPlotCollectionRegistry,
)
from artifact_core.binary_classification._types.plot_collections import (
    BinaryClassificationPlotCollectionType,
)

ThresholdVariationCurvesHyperparamsT = TypeVar(
    "ThresholdVariationCurvesHyperparamsT", bound="ThresholdVariationCurvesHyperparams"
)


@BinaryClassificationPlotCollectionRegistry.register_artifact_hyperparams(
    BinaryClassificationPlotCollectionType.THRESHOLD_VARIATION_CURVES
)
@dataclass(frozen=True)
class ThresholdVariationCurvesHyperparams(ArtifactHyperparams):
    curve_types: Sequence[ThresholdVariationCurveType]

    @classmethod
    def build(
        cls: Type[ThresholdVariationCurvesHyperparamsT],
        curve_types: Sequence[
            Union[ThresholdVariationCurveType, ThresholdVariationCurveTypeLiteral]
        ],
    ) -> ThresholdVariationCurvesHyperparamsT:
        ls_resolved = [
            curve_type
            if isinstance(curve_type, ThresholdVariationCurveType)
            else ThresholdVariationCurveType[curve_type]
            for curve_type in curve_types
        ]
        hyperparams = cls(curve_types=ls_resolved)
        return hyperparams


@BinaryClassificationPlotCollectionRegistry.register_artifact(
    BinaryClassificationPlotCollectionType.THRESHOLD_VARIATION_CURVES
)
class ThresholdVariationCurves(
    BinaryClassificationPlotCollection[ThresholdVariationCurvesHyperparams]
):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> Dict[str, Figure]:
        plot_collection = ThresholdVariationCurvePlotter.plot_multiple(
            curve_types=self._hyperparams.curve_types,
            true=true_class_store.id_to_is_positive,
            probs=classification_results.id_to_prob_pos,
        )
        result = {curve_type.value: plot for curve_type, plot in plot_collection.items()}
        return result
