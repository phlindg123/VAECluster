import torch
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.from_numpy(X)
        self.y = y
        if y is not None:
            self.y = torch.from_numpy(y)
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx].float(), self.y[idx].long()
        return self.X[idx].float()