from typing import Any, Mapping, Optional

import pytest
from artifact_core._utils.collections.map_merger import MapMerger


@pytest.mark.unit
@pytest.mark.parametrize(
    "base, override, expected",
    [
        # Simple merges
        ({"a": 1, "b": 2}, {"c": 3}, {"a": 1, "b": 2, "c": 3}),
        ({"a": 1}, {"a": 2}, {"a": 2}),
        ({"x": 10, "y": 20}, {}, {"x": 10, "y": 20}),
        ({}, {"a": 1}, {"a": 1}),
        ({}, {}, {}),
        # None override
        ({"a": 1, "b": 2}, None, {"a": 1, "b": 2}),
        # Nested dict merges
        (
            {"config": {"setting1": 1, "setting2": 2}},
            {"config": {"setting2": 20, "setting3": 30}},
            {"config": {"setting1": 1, "setting2": 20, "setting3": 30}},
        ),
        (
            {"level1": {"level2": {"value": 1}}},
            {"level1": {"level2": {"value": 10}}},
            {"level1": {"level2": {"value": 10}}},
        ),
        (
            {"nested": {"a": 1, "b": 2}},
            {"nested": {"b": 20}},
            {"nested": {"a": 1, "b": 20}},
        ),
        # Deep nesting
        (
            {"level1": {"level2": {"level3": {"value": 1, "another": 2}}}},
            {"level1": {"level2": {"level3": {"value": 100}}}},
            {"level1": {"level2": {"level3": {"value": 100, "another": 2}}}},
        ),
        # Override with non-dict replaces
        (
            {"config": {"setting1": 1, "setting2": 2}},
            {"config": "replaced"},
            {"config": "replaced"},
        ),
        # Base with non-dict gets replaced
        (
            {"config": "original"},
            {"config": {"setting1": 1}},
            {"config": {"setting1": 1}},
        ),
        # Lists are replaced, not merged
        ({"items": [1, 2, 3]}, {"items": [4, 5]}, {"items": [4, 5]}),
        # Complex scenario
        (
            {
                "database": {
                    "host": "localhost",
                    "port": 5432,
                    "credentials": {"user": "admin", "password": "secret"},
                },
                "cache": {"enabled": True, "ttl": 3600},
            },
            {
                "database": {
                    "port": 5433,
                    "credentials": {"password": "new_secret"},
                },
                "logging": {"level": "DEBUG"},
            },
            {
                "database": {
                    "host": "localhost",
                    "port": 5433,
                    "credentials": {"user": "admin", "password": "new_secret"},
                },
                "cache": {"enabled": True, "ttl": 3600},
                "logging": {"level": "DEBUG"},
            },
        ),
    ],
)
def test_merge(
    base: Mapping[str, Any],
    override: Optional[Mapping[str, Any]],
    expected: Mapping[str, Any],
):
    result = MapMerger.merge(base=base, override=override)
    assert result == expected
    assert isinstance(result, Mapping)
