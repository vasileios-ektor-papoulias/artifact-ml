from typing import Hashable, Mapping

import numpy as np


class BinaryDistributionCalculator:
    @classmethod
    def compute_id_to_probs(
        cls, id_to_prob_pos: Mapping[Hashable, float], pos_idx: int, eps: float = 1e-15
    ) -> Mapping[Hashable, np.ndarray]:
        id_to_probs: Mapping[Hashable, np.ndarray] = {}
        for identifier, prob_pos in id_to_prob_pos.items():
            prob_pos = cls._normalize_prob(prob_pos, eps=eps)
            arr = cls._build_probs_array(prob_pos=prob_pos, pos_idx=pos_idx)
            id_to_probs[identifier] = arr
        return id_to_probs

    @classmethod
    def compute_probs(cls, prob_pos: float, pos_idx: int, eps: float = 1e-15) -> np.ndarray:
        prob_pos = cls._normalize_prob(prob_pos, eps=eps)
        probs = cls._build_probs_array(prob_pos=prob_pos, pos_idx=pos_idx)
        return probs

    @classmethod
    def compute_id_to_prob_complement(
        cls, id_to_prob: Mapping[Hashable, float], eps: float = 1e-15
    ):
        id_to_probs = {
            identifier: cls.compute_prob_complement(prob=prob, eps=eps)
            for identifier, prob in id_to_prob.items()
        }
        return id_to_probs

    @classmethod
    def compute_prob_complement(cls, prob: float, eps: float) -> float:
        prob = cls._normalize_prob(prob=prob, eps=eps)
        prob_complement = cls._get_prob_complement(prob=prob)
        return prob_complement

    @staticmethod
    def _build_probs_array(prob_pos: float, pos_idx: int) -> np.ndarray:
        prob_neg = 1.0 - prob_pos
        arr = np.zeros(2, dtype=float)
        arr[pos_idx] = prob_pos
        arr[1 - pos_idx] = prob_neg
        return arr

    @classmethod
    def _get_prob_complement(cls, prob: float) -> float:
        prob_complement = 1.0 - prob
        return prob_complement

    @staticmethod
    def _normalize_prob(prob: float, eps: float) -> float:
        prob_normalized = float(np.clip(prob, eps, 1 - eps))
        return prob_normalized
