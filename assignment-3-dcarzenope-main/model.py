import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
  def __init__(self):
    super().__init__()                  
    self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)    
    self.pool = nn.MaxPool2d(2, 2)      
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
    self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.fc1 = nn.Linear(256*4*4,1024)
    self.fc2 = nn.Linear(1024,512)
    self.fc3 = nn.Linear(512,10)

    self.batchnorm1 = nn.BatchNorm2d(64)
    self.batchnorm2 = nn.BatchNorm2d(128)
    self.batchnorm3 = nn.BatchNorm2d(256)

  def forward(self, x):
    x = self.batchnorm1(self.pool(F.relu(self.conv2(F.relu(self.conv1(x))))))
    x = self.batchnorm2(self.pool(F.relu(self.conv4(F.relu(self.conv3(x))))))
    x = self.batchnorm3(self.pool(F.relu(self.conv6(F.relu(self.conv5(x))))))
    x = x.reshape(-1, 256*4*4)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)

    return x