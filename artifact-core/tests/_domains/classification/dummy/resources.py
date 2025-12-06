from artifact_core._domains.classification.resources import ClassificationArtifactResources
from artifact_core._libs.resources.classification.class_store import ClassStore
from artifact_core._libs.resources.classification.classification_results import (
    ClassificationResults,
)
from artifact_core._libs.resources.classification.distribution_store import ClassDistributionStore

from tests._domains.classification.dummy.resource_spec import DummyClassSpec

DummyClassStore = ClassStore[DummyClassSpec]
DummyDistributionStore = ClassDistributionStore[DummyClassSpec]
DummyClassificationResults = ClassificationResults[
    DummyClassSpec, DummyClassStore, DummyDistributionStore
]
DummyClassificationArtifactResources = ClassificationArtifactResources[
    DummyClassStore, DummyClassificationResults
]
