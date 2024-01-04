import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

class Model:
    def __init__(self, net, params, device):
        self.params = params
        self.device = device
        self.clf = net(**params).to(self.device)
        
    def fit(self, data):
        self.clf.train()
        
        n_epoch = self.params['epochs']
        optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])
        loader = DataLoader(data, shuffle=True, **self.params['train_args'])

        for epoch in (pbar := tqdm(range(1, n_epoch+1), ncols=100, desc="Epoch")):
            running_loss = .0
            for batch_idx, (x, y, _) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.clf(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()

                # Update running loss
                running_loss += loss.item()

            average_loss = running_loss / len(loader)
            pbar.set_description(f"Epoch {epoch} - Loss: {average_loss:.4f}")

    def predict(self, data):
        self.clf.eval()
        preds = torch.zeros((len(data),), dtype=torch.long)
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)
                pred = out.max(1)[1]
                preds[idxs] = pred.cpu()
        preds = preds.numpy()
        return preds
    
    def predict_prob(self, data):
        self.clf.eval()
        probs = torch.zeros([len(data), len(np.unique(data.target))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        probs = probs.numpy()
        return probs
    
    def predict_prob_dropout(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([len(data), len(np.unique(data.target))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        probs = probs.numpy()
        return probs
    
    def predict_prob_dropout_split(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([n_drop, len(data), len(np.unique(data.target))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[i][idxs] += prob.cpu()
        return probs
    
    def get_embeddings(self, data):
        self.clf.eval()
        embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                _, e1 = self.clf(x)
                embeddings[idxs] = e1.cpu()
        return embeddings