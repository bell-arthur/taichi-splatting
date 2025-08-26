import tinycudann as tcnn
import torch
import torch.nn as nn


class ConfigurableMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers=[32], activation='ReLU', use_hash_encoding=False, hash_config=None):
        super().__init__()
        self.use_hash_encoding = use_hash_encoding

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

    def _build_mlp(self, in_dim: int, out_dim: int, hidden_layers, activation_name: str) -> nn.Sequential:
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

        # Kaiming initialization for hidden layers (leaves output as default)
        for idx, module in enumerate(mlp):
            if isinstance(module, nn.Linear):
                # Use fan_in for ReLU-like activations
                nn.init.kaiming_uniform_(module.weight, a=0.01, nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        return mlp
