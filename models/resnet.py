import torch.nn as nn
import torch.nn.functional as F
from config import logger

class Block(nn.Module):

    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
    

class ResNet(nn.Module):
    def __init__(self, image_channels, num_classes, layers, **kwargs):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet18 architecture are in these 4 lines below
        self.layer1 = self.make_layer(Block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(Block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(Block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(Block, 512, layers[3], stride=2)

        self.avg_pool = nn.AvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                                       nn.BatchNorm2d(out_channels))

        layers = []
        layers.append(block(self.in_channels, out_channels, downsample, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out
    
    @property
    def get_embedding_dim(self):
        return 64