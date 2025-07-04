from dataclasses import dataclass
from typing import Dict

import pandas as pd

from artifact_core.base.artifact_dependencies import ArtifactHyperparams
from artifact_core.libs.implementation.tabular.js.calculator import JSDistanceCalculator
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonScoreCollection,
)
from artifact_core.table_comparison.registries.score_collections.registry import (
    TableComparisonScoreCollectionRegistry,
    TableComparisonScoreCollectionType,
)


@TableComparisonScoreCollectionRegistry.register_artifact_config(
    TableComparisonScoreCollectionType.JS_DISTANCE
)
@dataclass(frozen=True)
class JSDistanceScoresHyperparams(ArtifactHyperparams):
    n_bins_cts_histogram: int
    categorical_only: bool


@TableComparisonScoreCollectionRegistry.register_artifact(
    TableComparisonScoreCollectionType.JS_DISTANCE
)
class JSDistanceScores(TableComparisonScoreCollection[JSDistanceScoresHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Dict[str, float]:
        dict_js = JSDistanceCalculator.compute_dict_js(
            df_real=dataset_real,
            df_synthetic=dataset_synthetic,
            ls_cts_features=self._resource_spec.ls_cts_features,
            ls_cat_features=self._resource_spec.ls_cat_features,
            cat_unique_map=self._resource_spec.cat_unique_map,
            n_bins_cts_histogram=self._hyperparams.n_bins_cts_histogram,
            categorical_only=self._hyperparams.categorical_only,
        )
        return dict_js
