
import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from PIL import Image
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
        x = Image.fromarray(x.numpy(), mode='L')
        x = self.transforms(x)
        return x, y, index
        
def get_mnist_al_dataset(data_dir, num_valid):
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), data_dir)
    os.makedirs(data_path, exist_ok=True)

    mnist_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,)) # zero mean, unit std
    ])

    train_dataset = MNIST(root=data_path, train=True, target_transform=mnist_transform, download=True)
    test_dataset = MNIST(root=data_path, train=False, target_transform=mnist_transform, download=True)
    return Data(train_dataset, test_dataset, MNIST_Dataset, mnist_transform, num_valid)