import pytest


@pytest.mark.unit
def test_public_api_exports():
    import artifact_core.spi.artifact as module

    expected_all = [
        "Artifact",
        "NO_ARTIFACT_HYPERPARAMS",
        "ArtifactHyperparams",
        "ClassificationArtifact",
        "DatasetComparisonArtifact",
    ]
    assert hasattr(module, "__all__")
    assert set(module.__all__) == set(expected_all)
    for name in expected_all:
        assert hasattr(module, name), f"{name} not exported"
