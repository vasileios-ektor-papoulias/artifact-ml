import pandas as pd
from artifact_core.base.artifact_dependencies import NO_ARTIFACT_HYPERPARAMS, NoArtifactHyperparams
from artifact_core.libs.data_spec.tabular.protocol import TabularDataSpecProtocol
from artifact_core.libs.validation.table_validator import TableValidator
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonArtifact,
)
from pytest_mock import MockerFixture


class DummyTableComparisonArtifact(TableComparisonArtifact[float, NoArtifactHyperparams]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> float:
        _ = dataset_real
        _ = dataset_synthetic
        return 1.0


def test_resource_validation(
    mocker: MockerFixture,
    data_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
):
    patch_validate = mocker.patch.object(
        target=TableValidator, attribute="validate", wraps=TableValidator.validate
    )
    artifact = DummyTableComparisonArtifact(
        data_spec=data_spec, hyperparams=NO_ARTIFACT_HYPERPARAMS
    )
    real_validated, synthetic_validated = artifact._validate_datasets(
        dataset_real=df_real, dataset_synthetic=df_synthetic
    )
    expected_features = [
        feature
        for feature in data_spec.ls_features
        if feature in data_spec.ls_cat_features or feature in data_spec.ls_cts_features
    ]
    assert patch_validate.call_count == 2
    first_call_args = patch_validate.call_args_list[0][1]
    pd.testing.assert_frame_equal(first_call_args["df"], df_real)
    assert first_call_args["ls_features"] == expected_features
    assert first_call_args["ls_cat_features"] == data_spec.ls_cat_features
    assert first_call_args["ls_cts_features"] == data_spec.ls_cts_features
    second_call_args = patch_validate.call_args_list[1][1]
    pd.testing.assert_frame_equal(second_call_args["df"], df_synthetic)
    assert second_call_args["ls_features"] == expected_features
    assert second_call_args["ls_cat_features"] == data_spec.ls_cat_features
    assert second_call_args["ls_cts_features"] == data_spec.ls_cts_features
    assert real_validated is not None
    assert synthetic_validated is not None
    assert isinstance(real_validated, pd.DataFrame)
    assert isinstance(synthetic_validated, pd.DataFrame)
