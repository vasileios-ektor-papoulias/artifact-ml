from typing import Dict, List, Optional

import numpy as np


class JuxtapositionArrayCalculator:
    @classmethod
    def juxtapose_score_collections(
        cls,
        dict_scores_real: Dict[str, float],
        dict_scores_synthetic: Dict[str, float],
        ls_keys: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        if ls_keys is None:
            ls_keys = list(dict_scores_real.keys())
        result = {
            key: cls.juxtapose_scores(
                score_real=dict_scores_real[key],
                score_synthetic=dict_scores_synthetic[key],
            )
            for key in ls_keys
        }
        return result

    @staticmethod
    def juxtapose_scores(
        score_real: float,
        score_synthetic: float,
    ) -> np.ndarray:
        result = np.array([score_real, score_synthetic])
        return result
