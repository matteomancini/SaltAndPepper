import torch.nn as nn


class SimpleLSTMNetwork(nn.Module):

    def __init__(self, input_len, num_classes, hidden_units, hidden_layers=2):
        super(SimpleLSTMNetwork, self).__init__()
        assert hidden_layers > 0, "At least one hidden layer required"

        layer_list = [nn.LSTM(input_len, hidden_units, bidirectional=True, batch_first=True), ExtractTensor()]
        for i in range(hidden_layers-1):
            layer_list.extend([nn.Linear(2*hidden_units, 2*hidden_units), nn.ReLU()])
        layer_list.append(nn.Linear(2*hidden_units, num_classes))

        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.layers(x)


class ExtractTensor(nn.Module):
    def forward(self,x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        return tensor[:, -1, :]
