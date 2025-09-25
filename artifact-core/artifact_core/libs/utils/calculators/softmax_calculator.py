from typing import Dict, Hashable

import numpy as np
from scipy.special import softmax


class SoftmaxCalculator:
    @staticmethod
    def compute_probs(logits: np.ndarray) -> np.ndarray:
        if logits.size == 0:
            return np.empty_like(logits)
        return softmax(logits, axis=-1)

    @classmethod
    def compute_probs_multiple(
        cls, id_to_logits: Dict[Hashable, np.ndarray]
    ) -> Dict[Hashable, np.ndarray]:
        id_to_probs = {
            identifier: cls.compute_probs(logits=logits)
            for identifier, logits in id_to_logits.items()
        }
        return id_to_probs

    @staticmethod
    def compute_logits(probs: np.ndarray) -> np.ndarray:
        if probs.size == 0:
            return np.empty_like(probs)
        arr_logits = np.empty_like(probs)
        np.log(probs, out=arr_logits, where=probs > 0.0)
        arr_logits[probs <= 0.0] = -np.inf
        return arr_logits

    @classmethod
    def compute_logits_multiple(
        cls, id_to_probs: Dict[Hashable, np.ndarray]
    ) -> Dict[Hashable, np.ndarray]:
        id_to_logits = {
            identifier: cls.compute_logits(probs=probs) for identifier, probs in id_to_probs.items()
        }
        return id_to_logits
