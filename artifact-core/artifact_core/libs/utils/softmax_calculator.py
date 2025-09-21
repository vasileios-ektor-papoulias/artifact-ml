import numpy as np
from scipy.special import softmax


class SoftmaxCalculator:
    @staticmethod
    def compute_probs(logits: np.ndarray) -> np.ndarray:
        if logits.size == 0:
            return np.empty_like(logits)
        return softmax(logits, axis=-1)

    @staticmethod
    def compute_logits(probs: np.ndarray) -> np.ndarray:
        if probs.size == 0:
            return np.empty_like(probs)
        arr_logits = np.empty_like(probs)
        np.log(probs, out=arr_logits, where=probs > 0.0)
        arr_logits[probs <= 0.0] = -np.inf
        return arr_logits
