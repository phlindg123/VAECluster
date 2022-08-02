import torch
from torch import nn
from .layers import Dense, Latent


class VAE(nn.Module):
    def __init__(self, n_x, n_z):
        super().__init__()
        
        self.enc = nn.Sequential(
            Dense(n_x, n_x//2),
            Dense(n_x//2, n_x//2),
            Dense(n_x//2, n_x//4),
            Latent(n_x//4, n_z)
        )
        self.dec = nn.Sequential(
            Dense(n_z, n_x//4),
            Dense(n_x//4, n_x//2),
            Dense(n_x//2, n_x//2),
            Dense(n_x//2, n_x)
        )
    
    def encode(self, x):
        z, mu, log_var = self.enc(x)
        return z
    
    def decode(self, z):
        return self.dec(z)
    
    