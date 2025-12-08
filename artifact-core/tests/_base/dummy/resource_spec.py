from dataclasses import dataclass

from artifact_core._base.core.resource_spec import ResourceSpecProtocol


@dataclass
class DummyResourceSpec(ResourceSpecProtocol):
    scale: float
