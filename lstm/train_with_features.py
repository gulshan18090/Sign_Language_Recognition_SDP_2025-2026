import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

class FeatureDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.long)

def train_model(features, targets, model, epochs=10, batch_size=32, lr=1e-3):
    dataset = FeatureDataset(features, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(loader):.4f}")
    return model

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.fc(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_type', choices=['word_prob', 'image'], required=True)
    parser.add_argument('--features_path', required=True)
    parser.add_argument('--targets_path', required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    args = parser.parse_args()

    features = np.load(args.features_path)
    targets = np.load(args.targets_path)
    input_dim = features.shape[1]
    model = SimpleClassifier(input_dim, args.num_classes)
    train_model(features, targets, model)
