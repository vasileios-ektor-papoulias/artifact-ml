import pytest
from artifact_core._bootstrap.primitives.domain_toolkit import DomainToolkit


@pytest.mark.unit
@pytest.mark.parametrize(
    "toolkit, expected_name, expected_config_filename",
    [
        (
            DomainToolkit.TABLE_COMPARISON,
            "table_comparison",
            "table_comparison.json",
        ),
        (
            DomainToolkit.BINARY_CLASSIFICATION,
            "binary_classification",
            "binary_classification.json",
        ),
    ],
)
def test_domain_toolkit_properties(
    toolkit: DomainToolkit,
    expected_name: str,
    expected_config_filename: str,
):
    assert toolkit.toolkit_name == expected_name
    assert toolkit.config_override_filename == expected_config_filename
    package_root = toolkit.package_root
    assert package_root.is_dir()
    assert package_root.name == "artifact_core"
    toolkit_root = toolkit.toolkit_root
    assert toolkit_root == package_root / expected_name
    assert toolkit_root.is_dir()
    native_artifacts_dir = toolkit.native_artifacts_dir
    assert native_artifacts_dir == toolkit_root / "_artifacts"
    assert native_artifacts_dir.is_dir()
    base_config_filepath = toolkit.base_config_filepath
    assert base_config_filepath == toolkit_root / "_config" / "raw.json"
    assert base_config_filepath.is_file()
