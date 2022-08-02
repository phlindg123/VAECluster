import torch
from torch import nn

class Dense(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(i, o),
            nn.ELU()
        )
    def forward(self, x):
        return self.net(x)
    
class Latent(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.mu = nn.Linear(i, o)
        self.log_var = nn.Linear(i, o)
    
    def forward(self, x):
        mu = self.mu(x)
        log_var = self.log_var(x) + 1e-4
        sigma = log_var.exp()
        return mu + sigma * torch.randn_like(sigma), mu, log_var