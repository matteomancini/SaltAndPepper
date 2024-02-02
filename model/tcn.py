import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weight_norm
import math


class TCNBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, dilation, dropout, stride):
        super(TCNBlock, self).__init__()
        self.pad = ZeroPadding1d(math.floor((kernel_size - 1) * dilation / 2),
                                 math.ceil((kernel_size - 1) * dilation / 2))
        self.conv1 = weight_norm(nn.Conv1d(input_size, output_size, kernel_size, stride=stride, dilation=dilation))
        nn.init.xavier_normal_(self.conv1.weight)
        self.conv2 = weight_norm(nn.Conv1d(output_size, output_size, kernel_size, stride=stride, dilation=dilation))
        nn.init.xavier_normal_(self.conv2.weight)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.resample = nn.Conv1d(input_size, output_size, 1) if input_size != output_size else None
        if self.resample is not None:
            nn.init.xavier_normal_(self.resample.weight)
        self.final_relu = nn.ReLU()

    def forward(self, x):
        x_skip = x
        x = self.pad(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.pad(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        # in case input and output cannot be summed, adjust size with 1x1 conv
        res = x_skip if self.resample is None else self.resample(x_skip)
        x = x + res
        return self.final_relu(x)


class TCNAutoencoder(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=20, n_blocks=5, latent_size=8, stride=1,
                 t_len=1000, pool_step=32, dropout=0.1, final_layer='linear'):
        super(TCNAutoencoder, self).__init__()

        final_layer_dict = {'linear': nn.Linear(output_size, input_size),
                            'conv': weight_norm(nn.Conv1d(output_size, input_size, kernel_size=1, padding='same'))}
        if final_layer not in final_layer_dict.keys():
            raise ValueError(f"Argument 'final_layer' must be one of: {final_layer.keys()}")

        # Encoder
        encoder_layers = []
        # first block is input_size -> output_size, following ones are output_size -> output_size
        encoder_layers.append(TCNBlock(input_size, output_size, kernel_size=kernel_size,
                                       dilation=1, stride=stride, dropout=dropout))
        for n in range(1, n_blocks):
            encoder_layers.append(TCNBlock(output_size, output_size, kernel_size=kernel_size,
                                           dilation=2 ** n, stride=stride, dropout=dropout))
        self.encoder = nn.ModuleList(encoder_layers)

        # one last convolution on the latent data
        self.latent_layer = weight_norm(nn.Conv1d(output_size, latent_size, kernel_size=1, padding='same'))
        self.pool_step = pool_step
        # upsample layer to un-do pooling
        self.upsample = nn.Upsample(size=t_len)

        # Decoder
        decoder_layers = []
        # first block is latent_size -> output_size, following ones are output_size -> output_size
        decoder_layers.append(TCNBlock(latent_size, output_size, kernel_size=kernel_size,
                                       stride=stride, dilation=2 ** (n_blocks - 1), dropout=dropout))
        for n in reversed(range(n_blocks - 1)):
            decoder_layers.append(TCNBlock(output_size, output_size, kernel_size=kernel_size,
                                           stride=stride, dilation=2 ** n, dropout=dropout))
        self.decoder = nn.ModuleList(decoder_layers)

        # one final layer
        self.final_layer = final_layer_dict[final_layer]

    def forward(self, x):
        # Encoder
        for l in self.encoder:
            x = l(x)

        x = self.latent_layer(x)
        x = F.avg_pool1d(x, kernel_size=self.pool_step, stride=self.pool_step)
        x = self.upsample(x)

        # Decoder
        for l in self.decoder:
            x = l(x)

        x = x.permute(0, 2, 1)
        x = self.final_layer(x)
        x = x.permute(0, 2, 1)

        return x


# an alternative implementation of zeropadding
class ZeroPadding1d(nn.Module):
    def __init__(self, left, right):
        super(ZeroPadding1d, self).__init__()
        self.left = left
        self.right = right

    def forward(self, x):
        return F.pad(x, (self.left, self.right))