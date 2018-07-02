from typing import Any, Dict, List, Set, Tuple
from overrides import overrides

import torch


class MentionRecall(object):
    def __init__(self) -> None:
        self._num_gold_mentions = 0
        self._num_recalled_mentions = 0

    def __call__(self,  # type: ignore
                 top_spans: torch.Tensor,
                 gold_starts,
                 gold_ends):

        gold_mentions = {(gold[0], gold[1]) for _, gold in enumerate(zip(gold_starts, gold_ends))}

        predicted_spans = {(span[0], span[1]) for span in top_spans.data.tolist()}
        self._num_gold_mentions += len(gold_mentions)
        self._num_recalled_mentions += len(gold_mentions & predicted_spans)

    def get_metric(self, reset: bool = False) -> float:
        if self._num_gold_mentions == 0:
            recall = 0.0
        else:
            recall = self._num_recalled_mentions/float(self._num_gold_mentions)
        if reset:
            self.reset()
        return recall

    def reset(self):
        self._num_gold_mentions = 0
        self._num_recalled_mentions = 0
