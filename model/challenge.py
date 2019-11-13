'''
EECS 445 - Introduction to Machine Learning
Fall 2019 - Project 2
Challenge
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Challenge(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO:
        self.conv1 = nn.Conv2d(3,16,(5,5), (2,2), padding=2)
        self.conv2 = nn.Conv2d(16,64,(5,5), (2,2), padding=2)
        self.conv3 = nn.Conv2d(64,32,(5,5), (2,2), padding=2)
        self.fc1 = nn.Linear(512,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,5)
        

        self.init_weights()

    def init_weights(self):
        # TODO:
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5*5*C_in))
            nn.init.constant_(conv.bias, 0.0)

        nn.init.normal_(self.fc1.weight, 0.0, 1 / sqrt(512))
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.normal_(self.fc2.weight, 0.0, 1 / sqrt(64))
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.normal_(self.fc3.weight, 0.0, 1 / sqrt(32))
        nn.init.constant_(self.fc3.bias, 0.0)
        

    def forward(self, x):
        N, C, H, W = x.shape
        z = F.relu(self.conv1(x))
        z = F.relu(self.conv2(z))
        z = F.relu(self.conv3(z))
        z = z.view(N, 512)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = self.fc3(z)
        return z
