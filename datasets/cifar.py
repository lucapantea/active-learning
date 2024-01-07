import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from PIL import Image

from .data import Data

class CIFAR10_Dataset(Dataset):
    def __init__(self, data, target, transforms):
        self.data = data
        self.target = target
        self.transforms = transforms
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        x = Image.fromarray(x)
        x = self.transforms(x)
        return x, y, index

def get_cifar10_al_dataset(data_dir, num_valid, noise_transform=None, noise_rate=0.0):
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), data_dir)
    os.makedirs(data_path, exist_ok=True)

    cifar_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.247, 0.243, 0.261))
    ])

    if noise_transform is not None:
        cifar_transforms.transforms.append( 
            transforms.RandomApply([noise_transform], p=noise_rate)
        )
    
    train_dataset = CIFAR10(root=data_path, train=True, target_transform=cifar_transforms, download=True)
    test_dataset = CIFAR10(root=data_path, train=False, target_transform=cifar_transforms, download=True)

    return Data(train_dataset, test_dataset, CIFAR10_Dataset, cifar_transforms, num_valid)