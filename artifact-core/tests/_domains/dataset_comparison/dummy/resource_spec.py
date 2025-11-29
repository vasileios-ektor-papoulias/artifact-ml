from dataclasses import dataclass

from artifact_core._domains.dataset_comparison.artifact import ResourceSpecProtocol


@dataclass
class DummyResourceSpec(ResourceSpecProtocol):
    scale: float

