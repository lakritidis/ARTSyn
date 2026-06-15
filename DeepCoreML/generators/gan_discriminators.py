# Discriminator and Critic models for GANs

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    A typical GAN Discriminator is a binary classifier that outputs 0/1 (real/fake) values. It
    identifies fake samples (coming from the Generator) from real samples (coming from the dataset).
    As its performance improves during training, the Generator's performance also improves.
    """

    def __init__(self, architecture=(128, 128), input_dim=2, p=0.5, negative_slope=0.2):
        """
        A simple Discriminator, implemented as a typical fully-connected feed-forward network. As a binary
        classifier, it includes only one neuron in the output layer; its activation is the `Sigmoid` function.
        All the other dense layers use `LeakyReLU` as activation function for all neurons. Each dense layer
        is followed by a `Dropout` layer to prevent the model from over-fitting.

        Args:
            architecture: The architecture of the fully connected net; a tuple with the number of neurons per layer.
            input_dim: The dimensionality of the input (i.e. training) data.
            p: The probability that a weight is dropped at each training epoch - Passed to the Dropout layer.
            negative_slope: Controls the angle of the negative slope (used for negative inputs) - Passed to LeakyReLU.
        """
        super().__init__()

        seq = []
        dim = input_dim

        # The hidden layers:
        for features in architecture:
            seq += [nn.Linear(in_features=dim, out_features=features),
                    nn.Dropout(p=p),
                    nn.LeakyReLU(negative_slope=negative_slope)]

            dim = features

        # The output layer:
        seq += [nn.Linear(in_features=dim, out_features=1),
                nn.Sigmoid()]

        self.model = nn.Sequential(*seq)

    def forward(self, x):
        return self.model(x)

    def display(self):
        print(self.model)
        print(self.model.parameters())


class PackedDiscriminator(nn.Module):
    """
    A typical GAN Discriminator is a binary classifier that outputs 0/1 (real/fake) values. It
    identifies fake samples (coming from the Generator) from real samples (coming from the dataset).
    As its performance improves during training, the Generator's performance also improves.

    A packed Discriminator accepts multiple inputs at once in the form of concatenated vectors.
    In this way it alleviates the problem of mode collapse in GANs. The Generator remains the same.
    """

    def __init__(self, architecture=(128, 128), input_dim=2, pac=10, p=0.5, negative_slope=0.2):
        """
        A packed Discriminator, implemented as a typical fully-connected feed-forward network. As a binary
        classifier, it includes only one neuron in the output layer; its activation is the `Sigmoid` function.
        All the other dense layers use `LeakyReLU` as activation function for all neurons. Each dense layer
        is followed by a `Dropout` layer to prevent the model from over-fitting.

        `PackedDiscriminator` is implemented on the basis of `Discriminator`, but it also accepts a `pac`
        argument. The number of neurons (dimensionality) of the input layer is `pac * input_dim`.

        Args:
            architecture: The architecture of the fully connected net; a tuple with the number of neurons per layer.
            input_dim: The dimensionality of the input (i.e. training) data.
            pac: Number of samples to group together when applying the discriminator. Defaults to 10.
            p: The probability that a weight is dropped at each training epoch. Defaults to 0.3.
            negative_slope: Controls the angle of the negative slope (used for negative inputs). Defaults to 0.01.
        """
        super().__init__()

        seq = []
        dim = pac * input_dim
        self._pac = pac
        self._pac_dim = dim

        for lay in list(architecture):
            seq += [nn.Linear(dim, lay),
                    nn.LeakyReLU(negative_slope=negative_slope),
                    nn.Dropout(p=p)]
            dim = lay

        seq += [nn.Linear(dim, 1),
                nn.Sigmoid()]
        self._model = nn.Sequential(*seq)

    # def forward(self, x):
    #    return self.model(x)

    def display(self):
        print(self._model)
        print(self._model.parameters())

    def forward(self, x):
        """Apply the Discriminator to the `input_`."""
        assert x.size()[0] % self._pac == 0
        return self._model(x.view(-1, self._pac_dim))


class Critic(nn.Module):
    """Discriminator for ctGAN."""

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super().__init__()
        dim = input_dim * pac
        self._pac = pac
        self._pac_dim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [nn.Linear(dim, item), nn.LeakyReLU(0.2), nn.Dropout(0.5)]
            dim = item

        seq += [nn.Linear(dim, 1)]
        self._seq = nn.Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', lambda_=10):
        """Compute the gradient penalty. From the paper on improved WGAN training."""
        alpha = torch.rand(real_data.size(0) // self._pac, 1, 1, device=device)
        alpha = alpha.repeat(1, self._pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradients_view = gradients.view(-1, self._pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = (gradients_view ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, x):
        """Apply the Discriminator to the `input_`."""
        assert x.size()[0] % self._pac == 0
        return self._seq(x.view(-1, self._pac_dim))
