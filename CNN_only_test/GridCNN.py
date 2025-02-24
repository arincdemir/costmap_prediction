import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class GridCNN(nn.Module):
    def __init__(self):
        super(GridCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=5, padding=2), 
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            
            nn.Conv2d(128, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )

    def forward(self, x):
        return torch.sigmoid(self.conv_layers(x))