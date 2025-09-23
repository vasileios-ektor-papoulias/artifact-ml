from typing import Dict, Hashable, List, Mapping

from artifact_core.libs.implementation.binary_classification.score_distribution.partitioner import (
    BinarySamplePartitioner,
    BinarySampleSplit,
)
from artifact_core.libs.utils.dict_aligner import DictAligner


class ScoreDistributionSampler:
    @classmethod
    def get_sample(
        cls,
        id_to_is_pos: Mapping[Hashable, bool],
        id_to_prob_pos: Mapping[Hashable, float],
        split: BinarySampleSplit,
    ) -> List[float]:
        _, y_true_bin, y_prob = DictAligner.align(left=id_to_is_pos, right=id_to_prob_pos)
        arr = BinarySamplePartitioner.partition(y_true_bin=y_true_bin, y_prob=y_prob, split=split)
        return arr.tolist()

    @classmethod
    def get_dict_samples(
        cls,
        id_to_is_pos: Mapping[Hashable, bool],
        id_to_prob_pos: Mapping[Hashable, float],
        splits: List[BinarySampleSplit],
    ) -> Dict[BinarySampleSplit, List[float]]:
        dict_samples: Dict[BinarySampleSplit, List[float]] = {}
        for split in splits:
            sample = cls.get_sample(
                id_to_is_pos=id_to_is_pos, id_to_prob_pos=id_to_prob_pos, split=split
            )
            dict_samples[split] = sample
        return dict_samples
