from typing import Dict, Hashable, Tuple

import pytest

BinaryDataTuple = Tuple[Dict[Hashable, bool], Dict[Hashable, bool], Dict[Hashable, float]]


@pytest.fixture
def binary_data_balanced() -> BinaryDataTuple:
    id_to_is_pos: Dict[Hashable, bool] = {0: True, 1: True, 2: False, 3: False, 4: True, 5: False}
    id_to_pred_pos: Dict[Hashable, bool] = {0: True, 1: False, 2: False, 3: True, 4: True, 5: False}
    id_to_prob_pos: Dict[Hashable, float] = {0: 0.9, 1: 0.4, 2: 0.2, 3: 0.6, 4: 0.8, 5: 0.1}
    return id_to_is_pos, id_to_pred_pos, id_to_prob_pos


@pytest.fixture
def binary_data_imbalanced() -> BinaryDataTuple:
    id_to_is_pos: Dict[Hashable, bool] = {0: True, 1: False, 2: False, 3: False, 4: False}
    id_to_pred_pos: Dict[Hashable, bool] = {0: True, 1: False, 2: True, 3: False, 4: False}
    id_to_prob_pos: Dict[Hashable, float] = {0: 0.95, 1: 0.1, 2: 0.55, 3: 0.2, 4: 0.05}
    return id_to_is_pos, id_to_pred_pos, id_to_prob_pos


@pytest.fixture
def binary_data_all_positive() -> BinaryDataTuple:
    id_to_is_pos: Dict[Hashable, bool] = {0: True, 1: True, 2: True}
    id_to_pred_pos: Dict[Hashable, bool] = {0: True, 1: True, 2: False}
    id_to_prob_pos: Dict[Hashable, float] = {0: 0.9, 1: 0.8, 2: 0.3}
    return id_to_is_pos, id_to_pred_pos, id_to_prob_pos


@pytest.fixture
def binary_data_all_negative() -> BinaryDataTuple:
    id_to_is_pos: Dict[Hashable, bool] = {0: False, 1: False, 2: False}
    id_to_pred_pos: Dict[Hashable, bool] = {0: False, 1: True, 2: False}
    id_to_prob_pos: Dict[Hashable, float] = {0: 0.1, 1: 0.6, 2: 0.2}
    return id_to_is_pos, id_to_pred_pos, id_to_prob_pos


@pytest.fixture
def binary_data_perfect() -> BinaryDataTuple:
    id_to_is_pos: Dict[Hashable, bool] = {0: True, 1: True, 2: False, 3: False}
    id_to_pred_pos: Dict[Hashable, bool] = {0: True, 1: True, 2: False, 3: False}
    id_to_prob_pos: Dict[Hashable, float] = {0: 0.99, 1: 0.95, 2: 0.05, 3: 0.01}
    return id_to_is_pos, id_to_pred_pos, id_to_prob_pos


@pytest.fixture
def binary_data_dispatcher(request: pytest.FixtureRequest) -> BinaryDataTuple:
    valid_fixtures = [
        "binary_data_balanced",
        "binary_data_imbalanced",
        "binary_data_all_positive",
        "binary_data_all_negative",
        "binary_data_perfect",
    ]
    if request.param not in valid_fixtures:
        raise ValueError(f"Fixture {request.param} not found. Valid: {valid_fixtures}")
    return request.getfixturevalue(request.param)
