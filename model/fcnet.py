import torch.nn as nn


class FullyConnectedNetwork(nn.Module):

    def __init__(self, input_len, num_classes, hidden_units, hidden_layers=2):
        super(FullyConnectedNetwork, self).__init__()
        assert hidden_layers > 0, "At least one hidden layer required"

        layer_list = [nn.Flatten(), nn.Linear(input_len, hidden_units), nn.ReLU()]
        for i in range(hidden_layers-1):
            layer_list.extend([nn.Linear(hidden_units, hidden_units), nn.ReLU()])
        layer_list.append(nn.Linear(hidden_units, num_classes))

        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.layers(x)
