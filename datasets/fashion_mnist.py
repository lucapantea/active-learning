import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from PIL import Image

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
        x = Image.fromarray(x, mode='L')
        x = self.transforms(x)
        return x, y, index
    
def get_fashion_mnist_al_dataset(data_dir, num_valid, noise_transform=None, noise_rate=0.0):
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), data_dir)
        os.makedirs(data_path, exist_ok=True)

        fashion_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.286,), (0.353,))
        ])

        if noise_transform is not None:
            fashion_transforms.transforms.append( 
                transforms.RandomApply([noise_transform], p=noise_rate)
            )

        train_dataset = FashionMNIST(root=data_path, train=True, target_transform=fashion_transforms, download=True)
        test_dataset = FashionMNIST(root=data_path, train=False, target_transform=fashion_transforms, download=True)

        return Data(train_dataset, test_dataset, FashionMNIST_Dataset, fashion_transforms, num_valid)
