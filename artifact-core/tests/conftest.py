import matplotlib
import pytest
from matplotlib import pyplot as plt


@pytest.fixture
def set_agg_backend():
    matplotlib.use("Agg")


@pytest.fixture
def close_all_figs_after_test():
    yield
    plt.close("all")
