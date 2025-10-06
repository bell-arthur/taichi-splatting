import tinycudann as tcnn
import torch.nn as nn
from typing import List, Optional


class ConfigurableMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_layers: Optional[List[int]] = None, activation: str = 'ReLU', use_hash_encoding: bool = False, hash_config: Optional[dict] = None):
        super().__init__()
        self.use_hash_encoding = use_hash_encoding
        self.hash_config = None
        if hidden_layers is None:
            hidden_layers = [32]

        if use_hash_encoding:
            if hash_config is None:
                hash_config = {
                    "otype": "HashGrid",
                    "n_levels": 8,
                    "n_features_per_level": 2,
                    "log2_hashmap_size": 19,
                    "base_resolution": 16,
                    "per_level_scale": 2.0
                }

            self.hash_config = hash_config
            self.encoder = tcnn.Encoding(
                n_input_dims=2, encoding_config=hash_config)
            input_dim = self.encoder.n_output_dims
        else:
            self.encoder = None
            input_dim = in_dim

        self.mlp = self._build_mlp(input_dim, out_dim, hidden_layers, activation)

    def forward(self, x):
        if self.use_hash_encoding:
            x = self.encoder(x)  # type: ignore[misc]
            # Ensure dtype/device match the MLP parameters (tcnn often returns fp16)
            first_param = next(self.mlp.parameters())
            x = x.to(dtype=first_param.dtype, device=first_param.device)
        return self.mlp(x)

    def encode_only(self, x):
        """Return encoded inputs (useful for diagnostics/logging).
        Requires use_hash_encoding=True.
        """
        assert self.use_hash_encoding and self.encoder is not None
        y = self.encoder(x)  # type: ignore[misc]
        first_param = next(self.mlp.parameters())
        return y.to(dtype=first_param.dtype, device=first_param.device)

    @staticmethod
    def _get_activation(name: str):
        mapping = {
            'ReLU': nn.ReLU,
            'LeakyReLU': nn.LeakyReLU,
            'SiLU': nn.SiLU,
            'ELU': nn.ELU,
            'GELU': nn.GELU,
            'Tanh': nn.Tanh,
            'Sigmoid': nn.Sigmoid,
            'Softplus': nn.Softplus,
        }
        return mapping.get(name, nn.ReLU)

    def _build_mlp(self, in_dim: int, out_dim: int, hidden_layers: List[int], activation_name: str) -> nn.Sequential:
        layers = []
        last_dim = in_dim
        Act = self._get_activation(activation_name)

        if hidden_layers:
            for width in hidden_layers:
                layers.append(nn.Linear(last_dim, width))
                layers.append(Act())
                last_dim = width
        
        layers.append(nn.Linear(last_dim, out_dim))

        mlp = nn.Sequential(*layers)

        # Kaiming initialisation for hidden layers (leaves output as default)
        for idx, module in enumerate(mlp):
            if isinstance(module, nn.Linear):
                # Use fan_in for ReLU-like activations
                nn.init.kaiming_uniform_(module.weight, a=0.01, nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        return mlp
