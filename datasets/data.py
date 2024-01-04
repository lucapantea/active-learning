import numpy as np

class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, dataloader, transforms):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.dataloader = dataloader
        self.transforms = transforms
        self.n_pool = len(X_train)
        self.n_test = len(X_test)
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
    
    def get_test_data(self):
        return self.dataloader(self.X_test, self.Y_test, self.transforms)
    
    def get_train_data(self):
        return self.dataloader(self.X_train, self.Y_train, self.transforms)

    