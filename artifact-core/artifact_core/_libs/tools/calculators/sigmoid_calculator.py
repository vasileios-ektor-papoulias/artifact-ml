from typing import Hashable, Mapping

import numpy as np
from scipy.special import expit, logit


class SigmoidCalculator:
    @staticmethod
    def compute_prob(logit: float) -> float:
        return float(expit(logit))

    @classmethod
    def compute_probs_multiple(
        cls, id_to_logit: Mapping[Hashable, float]
    ) -> Mapping[Hashable, float]:
        id_to_prob = {
            identifier: cls.compute_prob(logit) for identifier, logit in id_to_logit.items()
        }
        return id_to_prob

    @staticmethod
    def compute_logit(prob: float, eps: float = 1e-15) -> float:
        prob = float(np.clip(prob, eps, 1 - eps))
        return float(logit(prob))

    @classmethod
    def compute_logits_multiple(
        cls, id_to_prob: Mapping[Hashable, float], eps: float = 1e-15
    ) -> Mapping[Hashable, float]:
        id_to_logit = {
            identifier: cls.compute_logit(prob, eps=eps) for identifier, prob in id_to_prob.items()
        }
        return id_to_logit
