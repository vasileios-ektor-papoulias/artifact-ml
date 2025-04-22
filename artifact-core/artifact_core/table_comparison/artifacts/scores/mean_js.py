from dataclasses import dataclass

import pandas as pd

from artifact_core.base.artifact_dependencies import ArtifactHyperparams
from artifact_core.libs.implementation.tabular.js.js import JSDistanceCalculator
from artifact_core.libs.resource_spec.tabular.protocol import (
    TabularDataSpecProtocol,
)
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonScore,
)
from artifact_core.table_comparison.registries.scores.registry import (
    TableComparisonScoreRegistry,
    TableComparisonScoreType,
)


@TableComparisonScoreRegistry.register_artifact_config(TableComparisonScoreType.MEAN_JS_DISTANCE)
@dataclass(frozen=True)
class MeanJSDistanceHyperparams(ArtifactHyperparams):
    n_bins_cts_histogram: int
    categorical_only: bool


@TableComparisonScoreRegistry.register_artifact(TableComparisonScoreType.MEAN_JS_DISTANCE)
class MeanJSDistance(TableComparisonScore[MeanJSDistanceHyperparams]):
    def __init__(
        self,
        resource_spec: TabularDataSpecProtocol,
        hyperparams: MeanJSDistanceHyperparams,
    ):
        self._resource_spec = resource_spec
        self._hyperparams = hyperparams

    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> float:
        mean_js_distance = JSDistanceCalculator.compute_mean_js(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            ls_cts_features=self._resource_spec.ls_cts_features,
            ls_cat_features=self._resource_spec.ls_cat_features,
            cat_unique_map=self._resource_spec.cat_unique_map,
            n_bins_cts_histogram=self._hyperparams.n_bins_cts_histogram,
            categorical_only=self._hyperparams.categorical_only,
        )
        return mean_js_distance
