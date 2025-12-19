import torch
from collections import defaultdict

class RelevanceAccumulator:
    def __init__(self):
        self.scores = defaultdict(lambda: None)
        self.count = 0

    def update(self, layer_name, relevance):
        # relevance: (B, C, H, W)
        score = relevance.abs().sum(dim=(0, 2, 3)).detach().cpu()

        if self.scores[layer_name] is None:
            self.scores[layer_name] = score.clone()
        else:
            self.scores[layer_name] += score

    def increment(self, batch_size):
        self.count += batch_size

    def get_normalized_scores(self):
        return {
            k: v / self.count for k, v in self.scores.items()
        }
