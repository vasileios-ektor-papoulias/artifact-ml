import pandas as pd
from artifact_core.base.artifact_dependencies import NO_ARTIFACT_HYPERPARAMS
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonScore,
)
from artifact_core.table_comparison.registries.scores.registry import (
    TableComparisonScoreRegistry,
)


@TableComparisonScoreRegistry.register_custom_artifact("CUSTOM_SCORE")
class CustomScore(TableComparisonScore[NO_ARTIFACT_HYPERPARAMS]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> float:
        _ = dataset_real
        _ = dataset_synthetic
        return 0
