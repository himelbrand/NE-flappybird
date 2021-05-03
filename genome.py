import torch
import torch.nn as nn
import numpy as np

IN_SIZE = 2
OUT_SIZE = 2
HIDDEN_SIZE = 4

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
    
