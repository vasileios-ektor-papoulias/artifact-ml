import pytest


@pytest.mark.unit
def test_public_api_exports():
    import artifact_core.shared.collections as module

    expected_all = [
        "ClassStore",
        "ClassificationResults",
        "ClassDistributionStore",
        "Deduplicator",
        "EntityStore",
        "MapAligner",
        "MapMerger",
        "SequenceConcatenator",
    ]
    assert hasattr(module, "__all__")
    assert set(module.__all__) == set(expected_all)
    for name in expected_all:
        assert hasattr(module, name), f"{name} not exported"
