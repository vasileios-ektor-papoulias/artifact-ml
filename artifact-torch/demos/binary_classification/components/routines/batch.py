from typing import Any, List, Optional

from artifact_experiment.tracking import TrackingClient
from artifact_torch.base.components.callbacks.batch import BatchCallback
from artifact_torch.base.components.routines.batch import BatchRoutine
from artifact_torch.libs.components.callbacks.batch.loss import BatchLossCallback

from demos.binary_classification.components.routines.protocols import (
    DemoModelInput,
    DemoModelOutput,
)
from demos.binary_classification.config.constants import BATCH_LOSS_PERIOD


class DemoBatchRoutine(BatchRoutine[DemoModelInput, DemoModelOutput]):
    @staticmethod
    def _get_batch_callbacks(
        tracking_client: Optional[TrackingClient],
    ) -> List[BatchCallback[DemoModelInput, DemoModelOutput, Any]]:
        _ = tracking_client
        return [BatchLossCallback(period=BATCH_LOSS_PERIOD, tracking_client=None)]
