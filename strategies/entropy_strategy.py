import numpy as np
import torch
from .strategy import Strategy

class EntropySampling(Strategy):
    def __init__(self, dataset, model, **kwargs):
        super(EntropySampling, self).__init__(dataset, model, **kwargs)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)
        entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=1)
        highest_entropy_idxs = torch.argsort(entropy, descending=True)[:n]
        return unlabeled_idxs[highest_entropy_idxs]
