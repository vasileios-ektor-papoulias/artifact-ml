import pandas as pd
from artifact_core._base.core.hyperparams import NO_ARTIFACT_HYPERPARAMS, NoArtifactHyperparams
from artifact_core._base.typing.artifact_result import Score
from artifact_core._libs.resource_specs.table_comparison.protocol import TabularDataSpecProtocol
from artifact_core._libs.validation.table_comparison.table_validator import TableValidator
from artifact_core.table_comparison._artifacts.base import (
    TableComparisonArtifact,
)
from pytest_mock import MockerFixture


class DummyTableComparisonArtifact(TableComparisonArtifact[NoArtifactHyperparams, Score]):
    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> float:
        _ = dataset_real
        _ = dataset_synthetic
        return 1.0


def test_resource_validation(
    mocker: MockerFixture,
    resource_spec: TabularDataSpecProtocol,
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
):
    patch_validate = mocker.patch.object(
        target=TableValidator, attribute="validate", wraps=TableValidator.validate
    )
    artifact = DummyTableComparisonArtifact(
        resource_spec=resource_spec, hyperparams=NO_ARTIFACT_HYPERPARAMS
    )
    real_validated, synthetic_validated = artifact._validate_datasets(
        dataset_real=df_real, dataset_synthetic=df_synthetic
    )
    expected_features = [
        feature
        for feature in resource_spec.features
        if feature in resource_spec.cat_features or feature in resource_spec.cts_features
    ]
    assert patch_validate.call_count == 2
    first_call_args = patch_validate.call_args_list[0][1]
    pd.testing.assert_frame_equal(first_call_args["df"], df_real)
    assert first_call_args["ls_features"] == expected_features
    assert first_call_args["ls_cat_features"] == resource_spec.cat_features
    assert first_call_args["ls_cts_features"] == resource_spec.cts_features
    second_call_args = patch_validate.call_args_list[1][1]
    pd.testing.assert_frame_equal(second_call_args["df"], df_synthetic)
    assert second_call_args["ls_features"] == expected_features
    assert second_call_args["ls_cat_features"] == resource_spec.cat_features
    assert second_call_args["ls_cts_features"] == resource_spec.cts_features
    assert real_validated is not None
    assert synthetic_validated is not None
    assert isinstance(real_validated, pd.DataFrame)
    assert isinstance(synthetic_validated, pd.DataFrame)
