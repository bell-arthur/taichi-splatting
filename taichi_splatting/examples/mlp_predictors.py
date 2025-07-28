import torch.nn as nn
import tinycudann as tcnn


class ConfigurableMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers=[32], activation='ReLU', use_hash_encoding=False, hash_config=None):
        super().__init__()
        self.use_hash_encoding = use_hash_encoding

        if use_hash_encoding:
            if hash_config is None:
                hash_config = {
                    "otype": "HashGrid",
                    "n_levels": 16,
                    "n_features_per_level": 2,
                    "log2_hashmap_size": 19,
                    "base_resolution": 16,
                    "per_level_scale": 2.0
                }

            self.encoder = tcnn.Encoding(n_input_dims=2, encoding_config=hash_config)
            input_dim = self.encoder.n_output_dims
        else:
            self.encoder = None
            input_dim = in_dim

        network_config = {
            "otype": "FullyFusedMLP",
            "activation": activation,
            "output_activation": "None",
            "n_neurons": hidden_layers[0] if hidden_layers else 32,
            "n_hidden_layers": len(hidden_layers)
        }

        self.mlp = tcnn.Network(n_input_dims=input_dim, n_output_dims=out_dim, network_config=network_config)

    def forward(self, x):
        if self.use_hash_encoding:
            x = self.encoder(x)
        return self.mlp(x)
