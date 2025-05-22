from typing import Mapping, Sequence

from artifact_experiment.libs.tracking.clear_ml.adapter import (
    ClearMLRunAdapter,
)


class ClearMLScoreReader:
    @classmethod
    def get_all_scores(
        cls, run: ClearMLRunAdapter
    ) -> Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]]:
        dict_all_scores = run.get_exported_scores()
        return dict_all_scores

    @classmethod
    def get_score_history(
        cls,
        dict_all_scores: Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]],
        score_path: str,
    ) -> Mapping[str, Mapping[str, Sequence[float]]]:
        try:
            dict_score_history = dict_all_scores[score_path]
            return dict_score_history
        except KeyError as e:
            raise e

    @classmethod
    def get_series_from_score_history(
        cls, score_history: Mapping[str, Mapping[str, Sequence[float]]], series_name: str
    ) -> Mapping[str, Sequence[float]]:
        try:
            dict_series_history = score_history[series_name]
            return dict_series_history
        except KeyError as e:
            raise e

    @classmethod
    def get_xvalues_from_score_series(
        cls, score_series: Mapping[str, Sequence[float]]
    ) -> Sequence[float]:
        xvalues = score_series["x"]
        return xvalues

    @classmethod
    def get_yvalues_from_score_series(
        cls, score_series: Mapping[str, Sequence[float]]
    ) -> Sequence[float]:
        yvalues = score_series["y"]
        return yvalues
