from artifact_core.libs.resource_spec.binary.protocol import BinaryFeatureSpecProtocol

from artifact_experiment.core.classification.callback_resources import (
    ClassificationCallbackResources,
)

BinaryClassificationCallbackResources = ClassificationCallbackResources[BinaryFeatureSpecProtocol]
