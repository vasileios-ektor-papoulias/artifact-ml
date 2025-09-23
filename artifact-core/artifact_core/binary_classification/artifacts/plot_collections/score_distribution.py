from dataclasses import dataclass
from typing import Dict, Sequence, Type, TypeVar, Union

from matplotlib.figure import Figure

from artifact_core.base.artifact_dependencies import ArtifactHyperparams
from artifact_core.binary_classification.artifacts.base import (
    BinaryClassificationPlotCollection,
)
from artifact_core.binary_classification.registries.plot_collections.registry import (
    BinaryClassificationPlotCollectionRegistry,
)
from artifact_core.binary_classification.registries.plot_collections.types import (
    BinaryClassificationPlotCollectionType,
)
from artifact_core.libs.implementation.binary_classification.score_distribution.partitioner import (
    BinarySampleSplit,
    BinarySampleSplitLiteral,
)
from artifact_core.libs.implementation.binary_classification.score_distribution.plotter import (
    ScoreDistributionPlotter,
)
from artifact_core.libs.resources.categorical.category_store.binary import (
    BinaryCategoryStore,
)
from artifact_core.libs.resources.classification.binary_classification_results import (
    BinaryClassificationResults,
)

ScoreDistributionPlotsHyperparamsT = TypeVar(
    "ScoreDistributionPlotsHyperparamsT",
    bound="ScoreDistributionPlotsHyperparams",
)


@BinaryClassificationPlotCollectionRegistry.register_artifact_hyperparams(
    BinaryClassificationPlotCollectionType.SCORE_DISTRIBUTION_PLOTS
)
@dataclass(frozen=True)
class ScoreDistributionPlotsHyperparams(ArtifactHyperparams):
    split_types: Sequence[BinarySampleSplit]

    @classmethod
    def build(
        cls: Type[ScoreDistributionPlotsHyperparamsT],
        split_types: Sequence[Union[BinarySampleSplit, BinarySampleSplitLiteral]],
    ) -> ScoreDistributionPlotsHyperparamsT:
        ls_resolved = [
            split_type
            if isinstance(split_type, BinarySampleSplit)
            else BinarySampleSplit[split_type]
            for split_type in split_types
        ]
        hyperparams = cls(split_types=ls_resolved)
        return hyperparams


@BinaryClassificationPlotCollectionRegistry.register_artifact(
    BinaryClassificationPlotCollectionType.SCORE_DISTRIBUTION_PLOTS
)
class ScoreDistributionPlots(BinaryClassificationPlotCollection[ScoreDistributionPlotsHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> Dict[str, Figure]:
        dict_plots = ScoreDistributionPlotter.plot_multiple(
            id_to_is_pos=true_category_store.id_to_is_positive,
            id_to_prob_pos=classification_results.id_to_prob_pos,
            splits=self._hyperparams.split_types,
        )
        result = {split_type.value: plot for split_type, plot in dict_plots.items()}
        return result
