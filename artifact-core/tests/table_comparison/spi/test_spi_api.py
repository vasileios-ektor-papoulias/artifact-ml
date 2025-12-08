import pytest


@pytest.mark.unit
def test_public_api_exports():
    import artifact_core.table_comparison.spi as module

    expected_all = [
        "TabularDataSpecProtocol",
        "TableComparisonArray",
        "TableComparisonArrayCollection",
        "TableComparisonArtifact",
        "TableComparisonPlot",
        "TableComparisonPlotCollection",
        "TableComparisonScore",
        "TableComparisonScoreCollection",
        "TableComparisonArtifactResources",
        "TableComparisonArrayCollectionRegistry",
        "TableComparisonArrayRegistry",
        "TableComparisonPlotCollectionRegistry",
        "TableComparisonPlotRegistry",
        "TableComparisonScoreCollectionRegistry",
        "TableComparisonScoreRegistry",
    ]
    assert hasattr(module, "__all__")
    assert set(module.__all__) == set(expected_all)
    for name in expected_all:
        assert hasattr(module, name), f"{name} not exported"

