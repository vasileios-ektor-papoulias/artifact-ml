from typing import Mapping, Optional, Sequence

import numpy as np


class ScoreJuxtapositionCalculator:
    @classmethod
    def juxtapose_score_collections(
        cls,
        scores_real: Mapping[str, float],
        scores_synthetic: Mapping[str, float],
        keys: Optional[Sequence[str]] = None,
    ) -> Mapping[str, np.ndarray]:
        if keys is None:
            keys = list(scores_real.keys())
        dict_arr_juxtaposition = {
            key: cls.juxtapose_scores(
                score_real=scores_real[key],
                score_synthetic=scores_synthetic[key],
            )
            for key in keys
        }
        return dict_arr_juxtaposition

    @staticmethod
    def juxtapose_scores(
        score_real: float,
        score_synthetic: float,
    ) -> np.ndarray:
        arr_juxtaposition = np.array([score_real, score_synthetic])
        return arr_juxtaposition
