from typing import Generator

import matplotlib
import pytest
from matplotlib import pyplot as plt


@pytest.fixture
def set_agg_backend() -> None:
    matplotlib.use("Agg")


@pytest.fixture
def close_all_figs_after_test() -> Generator[None, None, None]:
    yield
    plt.close("all")
