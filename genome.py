import torch
import torch.nn as nn
import numpy as np

IN_SIZE = 2
OUT_SIZE = 2
HIDDEN_SIZE = 4
DEV = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Bird(nn.Module):
    def __init__(self,pop_id):
        super(Bird, self).__init__()
        self.dna = nn.Sequential(
            nn.Linear(IN_SIZE,HIDDEN_SIZE,False),
            nn.Sigmoid(),
            nn.Linear(HIDDEN_SIZE,OUT_SIZE,False),
            nn.Softmax(dim=-1)
        )
        self.id = pop_id
        self.fitness = 0
        self.score = 0
    
    def forward(self, obs):
        x = self.dna(torch.tensor(obs).float())
        return torch.argmax(x)
    
class BirdRGB(nn.Module):
    def __init__(self,pop_id):
        super(BirdRGB, self).__init__()

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_w = conv2d_size_out(conv2d_size_out(288,kernel_size=4,stride=2),kernel_size=3,stride=1)
        conv_h = conv2d_size_out(conv2d_size_out(512,kernel_size=4,stride=2),kernel_size=3,stride=1)

        self.dna = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=4, stride=2, bias=False),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, bias=False),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Linear(conv_w*conv_h*16, 128, False),
            nn.ReLU(),
            nn.Linear(128, OUT_SIZE, False),
            nn.Softmax(dim=-1)
        )
        self.dna = self.dna.to(DEV)
        self.id = pop_id
        self.fitness = 0
        self.score = 0
    
    def forward(self, obs):
        obs = obs.to(DEV)
        x = self.dna(torch.tensor(obs).float())
        return torch.argmax(x)