import pytest


@pytest.mark.unit
def test_public_api_exports():
    import artifact_core.shared.interfaces as module

    expected_exports = ["Serializable"]
    for name in expected_exports:
        assert hasattr(module, name), f"{name} not exported"

