from enum import Enum
from typing import List, Literal

import numpy as np

BinarySampleSplitLiteral = Literal["NONE", "POSITIVE", "NEGATIVE"]


class BinarySampleSplit(Enum):
    NONE = "NONE"
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


class BinarySamplePartitioner:
    @staticmethod
    def partition(
        y_true_bin: List[bool],
        y_prob: List[float],
        split: BinarySampleSplit,
    ) -> np.ndarray:
        y_true_arr = np.array(y_true_bin, dtype=bool)
        y_prob_arr = np.array(y_prob, dtype=float)
        if split is BinarySampleSplit.NONE:
            return y_prob_arr
        elif split is BinarySampleSplit.POSITIVE:
            return y_prob_arr[y_true_arr]
        elif split is BinarySampleSplit.NEGATIVE:
            return y_prob_arr[~y_true_arr]
        else:
            raise ValueError(f"Unsupported split type: {split}")
