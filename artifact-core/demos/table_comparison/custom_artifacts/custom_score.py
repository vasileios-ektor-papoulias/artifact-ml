import pandas as pd
from artifact_core._base.core.hyperparams import NO_ARTIFACT_HYPERPARAMS
from artifact_core.table_comparison.spi import TableComparisonScore, TableComparisonScoreRegistry


@TableComparisonScoreRegistry.register_custom_artifact("CUSTOM_SCORE")
class CustomScore(TableComparisonScore[NO_ARTIFACT_HYPERPARAMS]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> float:
        _ = dataset_real
        _ = dataset_synthetic
        return 0
