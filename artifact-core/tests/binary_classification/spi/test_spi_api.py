import pytest


@pytest.mark.unit
def test_public_api_exports():
    import artifact_core.binary_classification.spi as module

    expected_all = [
        "BinaryClassSpecProtocol",
        "BinaryClassificationArray",
        "BinaryClassificationArrayCollection",
        "BinaryClassificationArtifact",
        "BinaryClassificationPlot",
        "BinaryClassificationPlotCollection",
        "BinaryClassificationScore",
        "BinaryClassificationScoreCollection",
        "BinaryClassificationArrayCollectionRegistry",
        "BinaryClassificationArrayRegistry",
        "BinaryClassificationPlotCollectionRegistry",
        "BinaryClassificationPlotRegistry",
        "BinaryClassificationScoreCollectionRegistry",
        "BinaryClassificationScoreRegistry",
        "BinaryClassificationArtifactResources",
    ]
    assert hasattr(module, "__all__")
    assert set(module.__all__) == set(expected_all)
    for name in expected_all:
        assert hasattr(module, name), f"{name} not exported"

