from dataclasses import dataclass

from artifact_torch.base.model.io import LossOutput, ModelInput
from artifact_torch.core.model.generative import GenerationParams


class DemoModelInput(ModelInput):
    pass


class DemoModelOutput(LossOutput):
    pass


@dataclass
class DemoGenerationParams(GenerationParams):
    n_records: int
    temperature: float
