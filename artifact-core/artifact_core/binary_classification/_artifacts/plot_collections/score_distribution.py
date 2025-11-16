from dataclasses import dataclass
from typing import Dict, Sequence, Type, TypeVar, Union

from matplotlib.figure import Figure

from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._libs.artifacts.binary_classification.score_distribution.partitioner import (
    BinarySampleSplit,
    BinarySampleSplitLiteral,
)
from artifact_core._libs.artifacts.binary_classification.score_distribution.plotter import (
    ScorePDFPlotter,
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

ScoreDistributionPlotsHyperparamsT = TypeVar(
    "ScoreDistributionPlotsHyperparamsT", bound="ScoreDistributionPlotsHyperparams"
)


@BinaryClassificationPlotCollectionRegistry.register_artifact_hyperparams(
    BinaryClassificationPlotCollectionType.SCORE_PDF_PLOTS
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
    BinaryClassificationPlotCollectionType.SCORE_PDF_PLOTS
)
class ScoreDistributionPlots(BinaryClassificationPlotCollection[ScoreDistributionPlotsHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> Dict[str, Figure]:
        dict_plots = ScorePDFPlotter.plot_multiple(
            id_to_is_pos=true_class_store.id_to_is_positive,
            id_to_prob_pos=classification_results.id_to_prob_pos,
            splits=self._hyperparams.split_types,
        )
        result = {split_type.value: plot for split_type, plot in dict_plots.items()}
        return result
