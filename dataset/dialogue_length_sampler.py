import numpy as np
from typing import Iterator, List
from torch.utils.data import Sampler


class DialogueLengthSampler(Sampler[int]):

    def __init__(self, dialogue_lengths: List[int]) -> None:
        super(DialogueLengthSampler, self).__init__()
        self.dialogue_lengths = np.array(dialogue_lengths)

    def __len__(self) -> int:
        return len(self.dialogue_lengths)

    def __iter__(self) -> Iterator[int]:
        yield from np.argsort(self.dialogue_lengths).tolist()
