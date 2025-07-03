from dataclasses import dataclass
from typing import Dict

import pandas as pd
from artifact_core.base.artifact_dependencies import ArtifactHyperparams
from artifact_core.libs.resource_spec.tabular.protocol import (
    TabularDataSpecProtocol,
)
from artifact_core.table_comparison.artifacts.base import (
    TableComparisonScoreCollection,
)
from artifact_core.table_comparison.registries.score_collections.registry import (
    TableComparisonScoreCollectionRegistry,
)


@TableComparisonScoreCollectionRegistry.register_custom_artifact_config("BANI")
@dataclass(frozen=True)
class BaniHyperparams(ArtifactHyperparams):
    bani_result: float


@TableComparisonScoreCollectionRegistry.register_custom_artifact("BANI")
class Bani(TableComparisonScoreCollection[BaniHyperparams]):
    def __init__(
        self,
        resource_spec: TabularDataSpecProtocol,
        hyperparams: BaniHyperparams,
    ):
        self._resource_spec = resource_spec
        self._hyperparams = hyperparams

    def _compare_datasets(
        self, dataset_real: pd.DataFrame, dataset_synthetic: pd.DataFrame
    ) -> Dict[str, float]:
        return {"res": self._hyperparams.bani_result}
