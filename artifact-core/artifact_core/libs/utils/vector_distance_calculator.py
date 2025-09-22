from enum import Enum
from typing import Literal

import numpy as np

VectorDistanceMetricLiteral = Literal["L2", "MAE", "RMSE", "COSINE_SIMILARITY"]


class VectorDistanceMetric(Enum):
    L2 = "l2"
    MAE = "mae"
    RMSE = "rmse"
    COSINE_SIMILARITY = "cosine_similarity"


class VectorDistanceCalculator:
    @staticmethod
    def compute(
        metric: VectorDistanceMetric,
        v_1: np.ndarray,
        v_2: np.ndarray,
    ) -> float:
        v_1 = v_1.flatten()
        v_2 = v_2.flatten()
        if metric == VectorDistanceMetric.L2:
            return np.linalg.norm(v_1 - v_2).item()
        elif metric == VectorDistanceMetric.MAE:
            return np.mean(np.abs(v_1 - v_2)).item()
        elif metric == VectorDistanceMetric.RMSE:
            return np.sqrt(np.mean((v_1 - v_2) ** 2).item())
        elif metric == VectorDistanceMetric.COSINE_SIMILARITY:
            norm_product = np.linalg.norm(v_1) * np.linalg.norm(v_2)
            if norm_product == 0:
                return np.nan
            return np.dot(v_1, v_2) / norm_product
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")
