import pandas as pd
import pytest
from artifact_core.core.dataset_comparison.artifact import DatasetComparisonArtifactResources
from artifact_experiment.table_comparison.resources import TableComparisonCallbackResources


@pytest.mark.parametrize(
    "real_data, synthetic_data",
    [
        (pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [3, 4]})),
        (pd.DataFrame({"x": [1.1, 2.2]}), pd.DataFrame({"x": [2.2, 3.3]})),
        (pd.DataFrame(), pd.DataFrame()),
    ],
)
def test_build(real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
    resources = TableComparisonCallbackResources.build(
        dataset_real=real_data, dataset_synthetic=synthetic_data
    )

    assert isinstance(resources, TableComparisonCallbackResources)
    assert isinstance(resources.artifact_resources, DatasetComparisonArtifactResources)
    assert isinstance(resources.artifact_resources.dataset_real, pd.DataFrame)
    assert isinstance(resources.artifact_resources.dataset_synthetic, pd.DataFrame)
    pd.testing.assert_frame_equal(resources.artifact_resources.dataset_real, real_data)
    pd.testing.assert_frame_equal(resources.artifact_resources.dataset_synthetic, synthetic_data)
