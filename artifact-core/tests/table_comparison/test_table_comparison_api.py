import pytest


@pytest.mark.unit
def test_public_api_exports():
    import artifact_core.table_comparison as module

    expected_all = [
        "TabularDataSpec",
        "TableComparisonEngine",
        "TableComparisonArrayCollectionType",
        "TableComparisonArrayType",
        "TableComparisonPlotCollectionType",
        "TableComparisonPlotType",
        "TableComparisonScoreCollectionType",
        "TableComparisonScoreType",
    ]
    assert hasattr(module, "__all__")
    assert set(module.__all__) == set(expected_all)
    for name in expected_all:
        assert hasattr(module, name), f"{name} not exported"
