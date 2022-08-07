import torch
from torch import nn
from .layers import Dense, Latent


class VAE(nn.Module):
    def __init__(self, n_x, n_z):
        super().__init__()
        
        #self.mse = nn.MSELoss()

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
    
    def kl_loss(self, mu, log_var):
        kl = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean(dim=0)
        return kl

    def rec_loss(self, x, x_hat):
        mse = (x - x_hat).pow(2).pow(0.5).sum(dim=1).mean(dim=0)
        #mse = nn.MSELoss(reduction="sum")
        #mse = mse(x_hat, x)
        return mse
    
    def forward(self, x, y=None, beta=1.0):
        x = x.flatten(1)
        z, mu, log_var = self.enc(x)
        x_hat = self.dec(z)
        rec_loss = self.rec_loss(x, x_hat)
        kl_loss = self.kl_loss(mu, log_var + 1e-4)
        return rec_loss + beta*kl_loss