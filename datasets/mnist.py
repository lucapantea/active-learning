
import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

from .data import Data

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
        
def get_mnist_al_dataset(data_dir):
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), data_dir)
    os.makedirs(data_path, exist_ok=True)

    mnist_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,)) # zero mean, unit std
    ])

    train_dataset = MNIST(root=data_path, train=True, target_transform=mnist_transform, download=True)
    test_dataset = MNIST(root=data_path, train=False, target_transform=mnist_transform, download=True)

    X_train = train_dataset.data.numpy()
    Y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy()
    Y_test = test_dataset.targets.numpy()

    return Data(X_train, Y_train, X_test, Y_test, MNIST_Dataset, mnist_transform)