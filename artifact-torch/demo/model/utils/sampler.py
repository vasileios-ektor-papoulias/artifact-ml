from enum import Enum, auto
from typing import List

import torch


class SamplingStrategy(Enum):
    RANDOM_SAMPLE = auto()
    ARGMAX = auto()


class LogitSampler:
    @staticmethod
    def sample(
        ls_t_logits: List[torch.Tensor],
        temperature: float = 1.0,
        strategy: SamplingStrategy = SamplingStrategy.RANDOM_SAMPLE,
    ) -> torch.Tensor:
        if strategy == SamplingStrategy.RANDOM_SAMPLE:
            return LogitSampler._sample_random(ls_t_logits, temperature)
        elif strategy == SamplingStrategy.ARGMAX:
            return LogitSampler._sample_argmax(ls_t_logits)
        else:
            raise ValueError(f"Unsupported sampling strategy: {strategy}")

    @staticmethod
    def _sample_random(
        ls_t_logits: List[torch.Tensor],
        temperature: float,
    ) -> torch.Tensor:
        samples = [
            torch.distributions.Categorical(logits=t_logits / temperature).sample().unsqueeze(1)
            for t_logits in ls_t_logits
        ]
        return torch.cat(samples, dim=1)

    @staticmethod
    def _sample_argmax(
        ls_t_logits: List[torch.Tensor],
    ) -> torch.Tensor:
        preds = [t_logits.argmax(dim=1, keepdim=True) for t_logits in ls_t_logits]
        return torch.cat(preds, dim=1)
