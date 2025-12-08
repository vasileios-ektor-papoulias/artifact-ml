from typing import Generator

import matplotlib
import matplotlib.pyplot as plt
import pytest


@pytest.fixture
def set_agg_backend() -> None:
    matplotlib.use("Agg")


@pytest.fixture
def close_all_figs_after_test() -> Generator[None, None, None]:
    yield
    plt.close("all")
