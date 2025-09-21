from typing import Any, List, Optional

from artifact_experiment.base.tracking.client import TrackingClient
from artifact_torch.base.components.callbacks.batch import BatchCallback
from artifact_torch.base.components.routines.batch import BatchRoutine
from artifact_torch.base.model.base import Model
from artifact_torch.libs.components.callbacks.batch.loss import (
    BatchLossCallback,
)
from demos.table_comparison.config.constants import (
    BATCH_LOSS_PERIOD,
)
from demos.table_comparison.model.io import TabularVAEInput, TabularVAEOutput


class DemoBatchRoutine(BatchRoutine[TabularVAEInput, TabularVAEOutput, Model]):
    @staticmethod
    def _get_batch_callbacks(
        tracking_client: Optional[TrackingClient],
    ) -> List[BatchCallback[TabularVAEInput, TabularVAEOutput, Model, Any]]:
        _ = tracking_client
        return [BatchLossCallback(period=BATCH_LOSS_PERIOD, tracking_client=None)]
