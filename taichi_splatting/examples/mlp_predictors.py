import torch.nn as nn


class CovarianceMLP(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, latent):
        return self.net(latent)


class AlphaMLP(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, latent):
        return self.net(latent)


class ConfigurableMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers=[32], activation='ReLU'):
        super().__init__()
        act_layer = getattr(nn, activation)
        layers = []
        prev_dim = in_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act_layer())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
