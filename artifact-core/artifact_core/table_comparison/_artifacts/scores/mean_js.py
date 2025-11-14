from dataclasses import dataclass

import pandas as pd

from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._base.typing.artifact_result import Score
from artifact_core._libs.artifacts.table_comparison.js.calculator import JSDistanceCalculator
from artifact_core.table_comparison._artifacts.base import TableComparisonScore
from artifact_core.table_comparison._registries.scores import TableComparisonScoreRegistry
from artifact_core.table_comparison._types.scores import TableComparisonScoreType


@TableComparisonScoreRegistry.register_artifact_hyperparams(
    TableComparisonScoreType.MEAN_JS_DISTANCE
)
@dataclass(frozen=True)
class MeanJSDistanceScoreHyperparams(ArtifactHyperparams):
    n_bins_cts_histogram: int
    categorical_only: bool


@TableComparisonScoreRegistry.register_artifact(TableComparisonScoreType.MEAN_JS_DISTANCE)
class MeanJSDistanceScore(TableComparisonScore[MeanJSDistanceScoreHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Score:
        mean_js_distance = JSDistanceCalculator.compute_mean_js(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            cts_features=self._resource_spec.cts_features,
            cat_features=self._resource_spec.cat_features,
            cat_unique_map=self._resource_spec.cat_unique_map,
            n_bins_cts_histogram=self._hyperparams.n_bins_cts_histogram,
            categorical_only=self._hyperparams.categorical_only,
        )
        return mean_js_distance
