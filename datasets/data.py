import torch
import numpy as np

class Data:
    def __init__(self, train_dataset, test_dataset, dataloader, transforms, num_valid):
        train_set, val_set = torch.utils.data.random_split(train_dataset, [len(train_dataset)-num_valid, num_valid])
        self.X_train = train_set.dataset.data.numpy()
        self.Y_train = train_set.dataset.targets.numpy()
        self.X_valid = val_set.dataset.data.numpy()
        self.Y_valid = val_set.dataset.targets.numpy()
        self.X_test = test_dataset.data.numpy()
        self.Y_test = test_dataset.targets.numpy()
        self.dataloader = dataloader
        self.transforms = transforms
        self.n_pool = len(self.X_train)
        self.n_test = len(self.X_test)
        self.n_valid = len(val_set)
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)

    def initialize_labels(self, num):
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True
    
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
    