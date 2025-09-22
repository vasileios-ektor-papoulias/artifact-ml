from dataclasses import dataclass
from typing import Dict, Sequence, Type, TypeVar, Union

from matplotlib.figure import Figure

from artifact_core.base.artifact_dependencies import ArtifactHyperparams
from artifact_core.binary_classification.artifacts.base import BinaryClassificationPlotCollection
from artifact_core.binary_classification.registries.plot_collections.registry import (
    BinaryClassificationPlotCollectionRegistry,
)
from artifact_core.binary_classification.registries.plot_collections.types import (
    BinaryClassificationPlotCollectionType,
)
from artifact_core.libs.implementation.binary_classification.threshold_variation.plotter import (
    ThresholdVariationCurvePlotter,
    ThresholdVariationCurveType,
    ThresholdVariationCurveTypeLiteral,
)
from artifact_core.libs.resources.categorical.category_store.binary import (
    BinaryCategoryStore,
)
from artifact_core.libs.resources.classification.binary_classification_results import (
    BinaryClassificationResults,
)

ThresholdVariationCurvesHyperparamsT = TypeVar(
    "ThresholdVariationCurvesHyperparamsT",
    bound="ThresholdVariationCurvesHyperparams",
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
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> Dict[str, Figure]:
        plot_collection = ThresholdVariationCurvePlotter.plot_multiple(
            curve_types=self._hyperparams.curve_types,
            true=true_category_store.id_to_category,
            probs=classification_results.id_to_prob_pos,
            pos_label=self._resource_spec.positive_category,
        )
        result = {curve_type.value: plot for curve_type, plot in plot_collection.items()}
        return result
