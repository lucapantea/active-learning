import torch
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from config import logger

class Data:
    def __init__(self, train_dataset, test_dataset, dataloader, transforms, num_valid):
        # Split the dataset indices into training and validation sets
        train_indices, val_indices = torch.utils.data.random_split(range(len(train_dataset)), [len(train_dataset) - num_valid, num_valid])

        def make_ndarray(x):
            if isinstance(x, np.ndarray):
                return x
            if isinstance(x, list):
                return np.array(x)
            return x.numpy()

        # Convert the entire dataset to numpy arrays
        full_data = make_ndarray(train_dataset.data)
        full_targets = make_ndarray(train_dataset.targets)
        
        # Stratified split for training and validation sets
        sss = StratifiedShuffleSplit(n_splits=1, test_size=num_valid, random_state=42)
        train_indices, val_indices = next(sss.split(full_data, full_targets))
        
        # Use indices to create training and validation sets
        self.X_train = full_data[train_indices]
        self.Y_train = full_targets[train_indices]
        self.X_valid = full_data[val_indices]
        self.Y_valid = full_targets[val_indices]
        
        # Convert test dataset
        self.X_test = make_ndarray(test_dataset.data)
        self.Y_test = make_ndarray(test_dataset.targets)

        self.dataloader = dataloader
        self.transforms = transforms

        self.n_pool = len(self.X_train)
        self.n_test = len(self.X_test)
        self.n_valid = num_valid
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)

    def initialize_labels(self, num):
        # Stratified sample for initial labeled data
        stratified_sampler = StratifiedShuffleSplit(n_splits=1, test_size=num, random_state=42)
        _, initial_labeled_indices = next(stratified_sampler.split(self.X_train, self.Y_train))
        self.labeled_idxs[initial_labeled_indices] = True
    
    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.dataloader(self.X_train[labeled_idxs], self.Y_train[labeled_idxs], self.transforms)
    
    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.dataloader(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs], self.transforms)
    
    def get_unlabeled_indices(self):
        return np.arange(self.n_pool)[~self.labeled_idxs]
    
    def get_train_data(self):
        return self.dataloader(self.X_train, self.Y_train, self.transforms)
    
    def get_validation_data(self):
        return self.dataloader(self.X_valid, self.Y_valid, self.transforms)
    
    def get_test_data(self):
        return self.dataloader(self.X_test, self.Y_test, self.transforms)
    
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'
    
class AddSaltAndPepperNoise(object):
    def __init__(self, rate=0.1):
        self.rate = rate
    
    def __call__(self, tensor):
        mask = torch.rand(tensor.size()) < self.rate
        tensor[mask] = torch.randint(0, 1, (1,))
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + f'(noise_rate={self.rate})'