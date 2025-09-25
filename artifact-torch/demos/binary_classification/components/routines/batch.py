from typing import Any, List, Optional

from artifact_experiment.base.tracking.client import TrackingClient
from artifact_torch.base.components.callbacks.batch import BatchCallback
from artifact_torch.base.components.routines.batch import BatchRoutine
from artifact_torch.base.model.base import Model
from artifact_torch.libs.components.callbacks.batch.loss import (
    BatchLossCallback,
)
from demos.binary_classification.config.constants import BATCH_LOSS_PERIOD
from demos.binary_classification.model.io import MLPClassifierInput, MLPClassifierOutput


class DemoBatchRoutine(BatchRoutine[MLPClassifierInput, MLPClassifierOutput, Model]):
    @staticmethod
    def _get_batch_callbacks(
        tracking_client: Optional[TrackingClient],
    ) -> List[BatchCallback[MLPClassifierInput, MLPClassifierOutput, Model, Any]]:
        _ = tracking_client
        return [BatchLossCallback(period=BATCH_LOSS_PERIOD, tracking_client=None)]
