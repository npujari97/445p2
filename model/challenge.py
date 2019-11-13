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

        #

    def forward(self, x):
        N, C, H, W = x.shape

        # TODO:

        #

        return z
