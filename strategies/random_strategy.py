import numpy as np
from .strategy import Strategy

class RandomSampling(Strategy):
    def __init__(self, dataset, model, **kwargs):
        super(RandomSampling, self).__init__(dataset, model, **kwargs)
    
    def query(self, n):
        return np.random.choice(self.dataset.get_unlabeled_indices(), n, replace=False)
    