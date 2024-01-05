import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, image_channels, num_classes, embedding_dim=84, feature_map_dim=256, **kwargs):
        super(LeNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.conv1 = nn.Conv2d(image_channels, 6 * image_channels, kernel_size=5)
        self.conv2 = nn.Conv2d(6 * image_channels, 16 * image_channels, kernel_size=5)

        # Default: 16 x 4 x 4 for MNIST
        self.fc1 = nn.Linear(feature_map_dim, 120 * image_channels)
        self.fc2 = nn.Linear(120 * image_channels, 84 * image_channels)
        self.fc3 = nn.Linear(84 * image_channels, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @property
    def get_embedding_dim(self):
        return self.embedding_dim
