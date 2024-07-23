import torch.nn as nn


class MLP(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_sizes: tuple or list,
            output_size: int,
            hidden_activation: nn.Module = nn.ReLU(),
            output_activation: nn.Module = nn.Identity(),
            layer_norm: bool = False,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = nn.LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if self.layer_norm and i < len(self.fcs) - 1:
                x = self.layer_norms[i](x)
            x = self.hidden_activation(x)
        preactivation = self.last_fc(x)
        return self.output_activation(preactivation)
