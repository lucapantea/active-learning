
import os
import numpy as np 
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

class MNIST_Dataset(Dataset):
    def __init__(self, data, target, transforms):
        self.data = data
        self.target = target
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        x = self.transforms(x)
        return x, y, index
    
class MNIST_AL_Dataset:
    def __init__(self, X_train, Y_train, X_test, Y_test, transforms):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.dataset = MNIST_Dataset
        self.n_pool = len(X_train)
        self.n_test = len(X_test)
        self.transforms = transforms

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
    
    def initialize_labels(self, num):
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True
    
    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.dataset(self.X_train[labeled_idxs], self.Y_train[labeled_idxs], self.transforms)
    
    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.dataset(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs], self.transforms)
    
    def get_unlabeled_indices(self):
        return np.arange(self.n_pool)[~self.labeled_idxs]
    
    def get_test_data(self):
        return self.dataset(self.X_test, self.Y_test, self.transforms)
    
    def get_train_data(self):
        return self.dataset(self.X_train, self.Y_train, self.transforms)
    
def get_mnist_al_dataset(data_dir):
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), data_dir)
    os.makedirs(data_path, exist_ok=True)

    mnist_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))    
    ])

    train_dataset = MNIST(root=data_path, train=True, target_transform=mnist_transform, download=True)
    test_dataset = MNIST(root=data_path, train=False, target_transform=mnist_transform, download=True)

    X_train = train_dataset.data.numpy()
    Y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy()
    Y_test = test_dataset.targets.numpy()

    return MNIST_AL_Dataset(X_train, Y_train, X_test, Y_test, mnist_transform)