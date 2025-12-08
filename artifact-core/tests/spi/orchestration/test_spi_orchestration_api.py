import pytest


@pytest.mark.unit
def test_public_api_exports():
    import artifact_core.spi.orchestration as module

    expected_all = [
        "ArtifactEngine",
        "ArtifactRegistry",
        "ArtifactType",
        "ClassificationEngine",
        "ClassificationArtifactRegistry",
        "DatasetComparisonEngine",
        "DatasetComparisonArtifactRegistry",
    ]
    assert hasattr(module, "__all__")
    assert set(module.__all__) == set(expected_all)
    for name in expected_all:
        assert hasattr(module, name), f"{name} not exported"
