from torch.utils.data import DataLoader
from torch import optim
import torch
import numpy as np
import pandas as pd

class Trainer:
    def __init__(self, model, train_data, test_data):
        self.model = model
        self.train_loader = DataLoader(train_data, batch_size=32)
        self.test_loader = DataLoader(test_data, batch_size=32)
    
        self.opt = optim.Adam(model.parameters(), lr=5e-4)
    
    def _train(self, beta):
        train_loss = 0
        for i, x in enumerate(self.train_loader):
            self.opt.zero_grad()
            if len(x) == 2:
                loss = self.model.forward(*x, beta=beta)
            else:
                loss = self.model.forward(x, beta=beta)
            loss.backward()
            self.opt.step()
            train_loss += loss.item()
        return train_loss
    
    def _test(self, beta):
        test_loss = 0
        with torch.no_grad():
            for i, x in enumerate(self.train_loader):
                if len(x) == 2:
                    loss = self.model.forward(*x, beta=beta)
                else:
                    loss = self.model.forward(x, beta=beta)
                test_loss += loss.item()
        return test_loss
    
    def fit(self, epochs, beta=1.0):
        losses = pd.DataFrame(index=list(range(epochs)), columns = ["Train", "Test"])
        for e in range(epochs):
            train_loss = self._train(beta)
            test_loss = self._test(beta)
            losses.loc[e, ["Train", "Test"]] = train_loss, test_loss
            print(f"Epoch: {e}, Train: {np.round(train_loss, 2)}, Test: {np.round(test_loss, 2)}")
        return losses