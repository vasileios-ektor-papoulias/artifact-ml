from typing import Dict, Hashable, Mapping, Sequence

from numpy import ndarray

from artifact_core._libs.artifacts.binary_classification.confusion.normalizer import (
    ConfusionMatrixNormalizationStrategy,
    ConfusionMatrixNormalizer,
)
from artifact_core._libs.artifacts.binary_classification.confusion.raw import (
    ConfusionMatrixCell,
    RawConfusionCalculator,
)


class NormalizedConfusionCalculator(RawConfusionCalculator):
    @classmethod
    def compute_normalized_confusion_matrix(
        cls,
        true: Mapping[Hashable, bool],
        predicted: Mapping[Hashable, bool],
        normalization: ConfusionMatrixNormalizationStrategy,
    ) -> ndarray:
        arr_cm = cls._compute_normalized_confusion_matrix(
            true=true,
            predicted=predicted,
            normalization=normalization,
        )
        return arr_cm

    @classmethod
    def compute_confusion_matrix_multiple_normalizations(
        cls,
        true: Mapping[Hashable, bool],
        predicted: Mapping[Hashable, bool],
        normalization_types: Sequence[ConfusionMatrixNormalizationStrategy],
    ) -> Dict[ConfusionMatrixNormalizationStrategy, ndarray]:
        dict_arr_cm = {
            norm: cls._compute_normalized_confusion_matrix(
                true=true, predicted=predicted, normalization=norm
            )
            for norm in normalization_types
        }
        return dict_arr_cm

    @classmethod
    def compute_dict_normalized_confusion_counts(
        cls,
        true: Mapping[Hashable, bool],
        predicted: Mapping[Hashable, bool],
        normalization: ConfusionMatrixNormalizationStrategy,
    ) -> Dict[ConfusionMatrixCell, float]:
        arr_cm = cls._compute_normalized_confusion_matrix(
            true=true,
            predicted=predicted,
            normalization=normalization,
        )
        tp, fp, tn, fn = cls._get_counts_from_matrix(arr_cm=arr_cm)
        dict_counts = cls._format_dict_counts(tp=tp, fp=fp, tn=tn, fn=fn)
        return dict_counts

    @classmethod
    def compute_normalized_confusion_count(
        cls,
        confusion_matrix_cell: ConfusionMatrixCell,
        true: Mapping[Hashable, bool],
        predicted: Mapping[Hashable, bool],
        normalization: ConfusionMatrixNormalizationStrategy,
    ) -> float:
        arr_cm = cls._compute_normalized_confusion_matrix(
            true=true,
            predicted=predicted,
            normalization=normalization,
        )
        tp, fp, tn, fn = cls._get_counts_from_matrix(arr_cm=arr_cm)
        count = cls._get_confusion_matrix_cell(
            confusion_matrix_cell=confusion_matrix_cell, tp=tp, fp=fp, tn=tn, fn=fn
        )
        return count

    @classmethod
    def _compute_normalized_confusion_matrix(
        cls,
        true: Mapping[Hashable, bool],
        predicted: Mapping[Hashable, bool],
        normalization: ConfusionMatrixNormalizationStrategy,
    ) -> ndarray:
        arr_cm = cls._compute_confusion_matrix(true=true, predicted=predicted)
        arr_cm_normalized = ConfusionMatrixNormalizer.normalize_cm(
            arr_cm=arr_cm, normalization=normalization
        )
        return arr_cm_normalized
