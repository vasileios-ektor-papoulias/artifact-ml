import pandas as pd

from artifact_experiment.core.dataset_comparison.callback_resources import (
    DatasetComparisonCallbackResources,
)

TableComparisonCallbackResources = DatasetComparisonCallbackResources[pd.DataFrame]
