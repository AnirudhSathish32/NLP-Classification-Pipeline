import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score


class TorchTextClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_size=128,
        epochs=10,
        batch_size=32,
        lr=0.001,
        random_state=42
    ):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state

    def _build_model(self, input_size):
        return nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2)
        )

    def fit(self, X, y):
        torch.manual_seed(self.random_state)

        X = X.toarray()
        y = np.array(y)

        self.input_size_ = X.shape[1]
        self.model_ = self._build_model(self.input_size_)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X):
        X = torch.tensor(X.toarray(), dtype=torch.float32)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X)
            probs = torch.softmax(logits, dim=1)

        return probs.numpy()

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def score(self, X, y):
        probs = self.predict_proba(X)[:, 1]
        return roc_auc_score(y, probs)