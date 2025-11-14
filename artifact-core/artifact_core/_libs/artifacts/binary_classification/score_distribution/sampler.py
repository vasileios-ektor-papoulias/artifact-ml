from typing import Dict, Hashable, Mapping, Sequence

from artifact_core._libs.artifacts.binary_classification.score_distribution.partitioner import (
    BinarySamplePartitioner,
    BinarySampleSplit,
)
from artifact_core._utils.collections.map_aligner import MapAligner


class ScoreDistributionSampler:
    @classmethod
    def get_sample(
        cls,
        id_to_is_pos: Mapping[Hashable, bool],
        id_to_prob_pos: Mapping[Hashable, float],
        split: BinarySampleSplit,
    ) -> Sequence[float]:
        _, y_true_bin, y_prob = MapAligner.align(left=id_to_is_pos, right=id_to_prob_pos)
        arr = BinarySamplePartitioner.partition(
            y_true_bin=list(y_true_bin), y_prob=list(y_prob), split=split
        )
        return arr.tolist()  # type: ignore

    @classmethod
    def get_dict_samples(
        cls,
        id_to_is_pos: Mapping[Hashable, bool],
        id_to_prob_pos: Mapping[Hashable, float],
        splits: Sequence[BinarySampleSplit],
    ) -> Mapping[BinarySampleSplit, Sequence[float]]:
        dict_samples: Dict[BinarySampleSplit, Sequence[float]] = {}
        for split in splits:
            sample = cls.get_sample(
                id_to_is_pos=id_to_is_pos, id_to_prob_pos=id_to_prob_pos, split=split
            )
            dict_samples[split] = sample
        return dict_samples
