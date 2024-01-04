import os
from torch.utils.data import Dataset
from trochvision.datasets import FashionMNIST

from .data import Data

class FashionMNIST_Dataset(Dataset):
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
    
def get_fashion_mnist_al_dataset(data_dir):
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), data_dir)
        os.makedirs(data_path, exist_ok=True)

        transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.286,), (0.353,))
        ])

        train_dataset = FashionMNIST(root=data_path, train=True, target_transform=transforms, download=True)
        test_dataset = FashionMNIST(root=data_path, train=False, target_transform=transforms, download=True)

        X_train = train_dataset.data.numpy()
        Y_train = train_dataset.targets.numpy()
        X_test = test_dataset.data.numpy()
        Y_test = test_dataset.targets.numpy()

        return Data(X_train, Y_train, X_test, Y_test, FashionMNIST_Dataset, transforms)
