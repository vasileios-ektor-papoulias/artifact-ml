import matplotlib
import matplotlib.pyplot as plt
import pytest


@pytest.fixture
def set_agg_backend():
    matplotlib.use("Agg")


@pytest.fixture
def close_all_figs_after_test():
    yield
    plt.close("all")
