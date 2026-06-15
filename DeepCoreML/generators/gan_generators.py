# Generator models for GANs

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    The Generator is the part of the GAN that ultimately produces artificial data.
    """
    def __init__(self, architecture=(128, 128), input_dim=2, output_dim=2, activation='Sigmoid',
                 normalize=False, negative_slope=0.01):
        """
        A simple GAN Generator, implemented as a fully-connected feed-forward network with a stack
        of residual blocks. Leaky ReLU is the activation function of all neurons in each dense layer.
        Dropout is not applied here.

        Args:
            architecture: The architecture of the fully connected net; a tuple with the number of
                neurons per layer.
            input_dim: The dimensionality of the input (i.e. training) data.
            output_dim: The dimensionality of the data that will be generated; in most cases, equal
                to `input_dim`.
            activation: Determines the activation function of the output layer.
            normalize: If True, appends a 1-D Batch Normalization layer after each dense layer.
            negative_slope: Controls the angle of the negative slope (used for negative inputs);
                Passed to `LeakyReLU`.
        """
        super().__init__()

        def residual_block(in_dim, out_dim, norm, slope):
            layers = [nn.Linear(in_features=in_dim, out_features=out_dim)]

            # The results with BatchNorm enabled do not seem very good
            if norm:
                layers.append(nn.BatchNorm1d(out_dim))

            layers.append(nn.LeakyReLU(negative_slope=slope))

            return layers

        seq = []
        dim = input_dim
        for features in architecture:
            seq += [*residual_block(dim, features, normalize, negative_slope)]
            dim = features

        # Apply the specified activation function.
        if activation == 'tanh':
            seq += [nn.Linear(dim, output_dim), nn.Tanh()]
        elif activation == 'sigmoid':
            seq += [nn.Linear(dim, output_dim), nn.Sigmoid()]
        else:
            # Output layer with no activation; it just returns the weighted sum of the inputs
            seq += [nn.Linear(dim, output_dim)]

        self.model = nn.Sequential(*seq)

    def forward(self, x):
        return self.model(x)

    def display(self):
        print(self.model)
        print(self.model.parameters())


class Residual(nn.Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = nn.Linear(i, o)
        self.bn = nn.BatchNorm1d(o)
        self.relu = nn.ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class ctGenerator(nn.Module):
    """Generator for ctGAN and ctdGAN"""

    def __init__(self, embedding_dim, architecture, data_dim):
        super().__init__()
        dim = embedding_dim
        seq = []
        for item in list(architecture):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(nn.Linear(dim, data_dim))
        self.seq = nn.Sequential(*seq)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data
