from typing import Dict, List

import matplotlib
import numpy as np
import pytest
from artifact_core._base.artifact_dependencies import ArtifactResult
from matplotlib.figure import Figure

matplotlib.use("Agg")


@pytest.fixture
def standard_uuid_length() -> int:
    return 36


@pytest.fixture
def artifact_result(request) -> ArtifactResult:
    return request.getfixturevalue(request.param)


@pytest.fixture
def score(request) -> float:
    return request.getfixturevalue(request.param)


@pytest.fixture
def array(request) -> np.ndarray:
    return request.getfixturevalue(request.param)


@pytest.fixture
def plot(request) -> Figure:
    return request.getfixturevalue(request.param)


@pytest.fixture
def score_collection(request) -> Dict[str, float]:
    return request.getfixturevalue(request.param)


@pytest.fixture
def array_collection(request) -> Dict[str, np.ndarray]:
    return request.getfixturevalue(request.param)


@pytest.fixture
def plot_collection(request) -> Dict[str, Figure]:
    return request.getfixturevalue(request.param)


@pytest.fixture
def ls_artifact_results(request) -> List[ArtifactResult]:
    return [request.getfixturevalue(name) for name in request.param]


@pytest.fixture
def ls_scores(request) -> List[float]:
    return [request.getfixturevalue(name) for name in request.param]


@pytest.fixture
def ls_arrays(request) -> List[np.ndarray]:
    return [request.getfixturevalue(name) for name in request.param]


@pytest.fixture
def ls_plots(request) -> List[Figure]:
    return [request.getfixturevalue(name) for name in request.param]


@pytest.fixture
def ls_score_collections(request) -> List[Dict[str, float]]:
    return [request.getfixturevalue(name) for name in request.param]


@pytest.fixture
def ls_array_collections(request) -> List[Dict[str, np.ndarray]]:
    return [request.getfixturevalue(name) for name in request.param]


@pytest.fixture
def ls_plot_collections(request) -> List[Dict[str, Figure]]:
    return [request.getfixturevalue(name) for name in request.param]


@pytest.fixture
def score_1() -> float:
    return 0.85


@pytest.fixture
def score_2() -> float:
    return 0.92


@pytest.fixture
def score_3() -> float:
    return 0.73


@pytest.fixture
def score_4() -> float:
    return 0.68


@pytest.fixture
def score_5() -> float:
    return 0.95


@pytest.fixture
def array_1() -> np.ndarray:
    return np.array([1, 2, 3, 4, 5])


@pytest.fixture
def array_2() -> np.ndarray:
    return np.array([10, 20, 30])


@pytest.fixture
def array_3() -> np.ndarray:
    return np.array([0.1, 0.2, 0.3, 0.4])


@pytest.fixture
def array_4() -> np.ndarray:
    return np.array([100, 200, 300, 400, 500])


@pytest.fixture
def array_5() -> np.ndarray:
    return np.array([1.1, 2.2, 3.3])


@pytest.fixture
def plot_1() -> Figure:
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.plot([1, 2, 3], [4, 5, 6])
    return fig


@pytest.fixture
def plot_2() -> Figure:
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.scatter([1, 2, 3], [3, 1, 4])
    return fig


@pytest.fixture
def plot_3() -> Figure:
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.bar([1, 2, 3], [5, 7, 3])
    return fig


@pytest.fixture
def plot_4() -> Figure:
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.hist([1, 2, 2, 3, 3, 3], bins=3)
    return fig


@pytest.fixture
def plot_5() -> Figure:
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.boxplot([1, 2, 3, 4, 5])
    return fig


@pytest.fixture
def score_collection_1() -> Dict[str, float]:
    return {"accuracy": 0.95, "precision": 0.87, "recall": 0.92}


@pytest.fixture
def score_collection_2() -> Dict[str, float]:
    return {"f1": 0.89, "specificity": 0.94}


@pytest.fixture
def score_collection_3() -> Dict[str, float]:
    return {"mse": 0.05, "mae": 0.03, "rmse": 0.22}


@pytest.fixture
def score_collection_4() -> Dict[str, float]:
    return {"auc": 0.96, "pr_auc": 0.88}


@pytest.fixture
def score_collection_5() -> Dict[str, float]:
    return {"sensitivity": 0.91, "specificity": 0.89, "npv": 0.93}


@pytest.fixture
def array_collection_1() -> Dict[str, np.ndarray]:
    return {
        "predictions": np.array([1, 0, 1, 1, 0]),
        "targets": np.array([1, 0, 1, 0, 1]),
    }


@pytest.fixture
def array_collection_2() -> Dict[str, np.ndarray]:
    return {
        "features": np.array([1.0, 2.0, 3.0]),
        "weights": np.array([0.1, 0.2, 0.3]),
    }


@pytest.fixture
def array_collection_3() -> Dict[str, np.ndarray]:
    return {
        "train_loss": np.array([0.8, 0.6, 0.4]),
        "val_loss": np.array([0.9, 0.7, 0.5]),
    }


@pytest.fixture
def array_collection_4() -> Dict[str, np.ndarray]:
    return {
        "embeddings": np.array([0.1, 0.2, 0.3, 0.4]),
        "labels": np.array([0, 1, 0, 1]),
    }


@pytest.fixture
def array_collection_5() -> Dict[str, np.ndarray]:
    return {
        "probabilities": np.array([0.9, 0.1, 0.8, 0.2]),
        "confidence": np.array([0.95, 0.85, 0.9, 0.8]),
    }


@pytest.fixture
def plot_collection_1() -> Dict[str, Figure]:
    fig1 = Figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot([1, 2], [1, 2])

    fig2 = Figure()
    ax2 = fig2.add_subplot(111)
    ax2.scatter([1, 2], [1, 2])

    return {"line_plot": fig1, "scatter_plot": fig2}


@pytest.fixture
def plot_collection_2() -> Dict[str, Figure]:
    fig1 = Figure()
    ax1 = fig1.add_subplot(111)
    ax1.bar([1, 2, 3], [4, 5, 6])

    fig2 = Figure()
    ax2 = fig2.add_subplot(111)
    ax2.hist([1, 2, 2, 3, 3, 3], bins=3)

    return {"bar_plot": fig1, "histogram": fig2}


@pytest.fixture
def plot_collection_3() -> Dict[str, Figure]:
    fig1 = Figure()
    ax1 = fig1.add_subplot(111)
    ax1.boxplot([1, 2, 3, 4, 5])

    fig2 = Figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot([1, 2, 3], [3, 1, 4])

    return {"box_plot": fig1, "trend_plot": fig2}


@pytest.fixture
def plot_collection_4() -> Dict[str, Figure]:
    fig1 = Figure()
    ax1 = fig1.add_subplot(111)
    ax1.scatter([1, 2, 3, 4], [4, 3, 2, 1])

    return {"scatter_plot": fig1}


@pytest.fixture
def plot_collection_5() -> Dict[str, Figure]:
    fig1 = Figure()
    ax1 = fig1.add_subplot(111)
    ax1.pie([1, 2, 3, 4], labels=["A", "B", "C", "D"])

    fig2 = Figure()
    ax2 = fig2.add_subplot(111)
    ax2.step([1, 2, 3, 4], [1, 3, 2, 4])

    fig3 = Figure()
    ax3 = fig3.add_subplot(111)
    ax3.errorbar([1, 2, 3], [1, 2, 3], yerr=[0.1, 0.2, 0.1])

    return {"pie_chart": fig1, "step_plot": fig2, "error_plot": fig3}
