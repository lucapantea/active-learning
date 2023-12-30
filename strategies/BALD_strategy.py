import numpy as np
import torch
from .strategy import Strategy

class BALDDropout(Strategy):
    def __init__(self, dataset, model, n_drop=10, **kwargs):
        super(BALDDropout, self).__init__(dataset, model, **kwargs)
        self.n_drop = n_drop

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob_dropout_split(unlabeled_data, n_drop=self.n_drop)
        expected_entropy = -torch.sum(probs.mean(0) * torch.log(probs.mean(0) + 1e-5), dim=1)
        average_entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=2).mean(0)

        # BALD score: difference between the two entropies
        bald_scores = average_entropy - expected_entropy

        selected_idxs = torch.argsort(bald_scores, descending=True)[:n]
        return unlabeled_idxs[selected_idxs]