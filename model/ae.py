import torch.nn as nn


class SimpleAutoEncoder(nn.Module):
    def __init__(self, input_size, n_layers):
        super(SimpleAutoEncoder, self).__init__()

        encoder_layers = [l for n in range(n_layers) for l in
                          [nn.Linear(input_size // (2**n), input_size // (2**(n+1))), nn.ReLU()]]
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = [l for n in range(n_layers) for l in
                          [nn.Linear(input_size // (2**(n+1)), input_size // (2**n)), nn.ReLU()]]
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
