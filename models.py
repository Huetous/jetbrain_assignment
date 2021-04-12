import torch
import math
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def linear_layer(in_channels, out_channels):
    """
    Creates a block with a linear layer and an activation
    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :return: created block
    """
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.ReLU()
    )


class MLP(nn.Module):
    """
    Multilayer perceptron
    """

    def __init__(self, in_channels, out_channels, n_filters, include_head=True):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param n_filters: list that consists of number of filters for each layer
        :param include_head: add the head of MLP or not
        """
        super().__init__()
        n_filters = [in_channels] + n_filters

        layers = [
            linear_layer(n_filters[i], n_filters[i + 1])
            for i in range(len(n_filters) - 1)
        ]

        if include_head:
            layers += [nn.Linear(n_filters[-1], out_channels)]

        self.model = nn.Sequential(*layers).to(device)

    def forward(self, x):
        return self.model(x)


# Since modules V and A both propagate their error to the last convolutional
# layer we rescale gradients (paper - https://arxiv.org/pdf/1511.06581.pdf)
def scale_gradients(module, grad_out, grad_in):
    return tuple(map(lambda grad: grad / math.sqrt(2.0), grad_out))


class DuelingMLP(nn.Module):
    """
    For more detailed explanation, please refer to https://arxiv.org/pdf/1511.06581.pdf
    """

    def __init__(self, in_channels, out_channels, n_filters):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param n_filters: list that consists of number of filters for each layer
        """
        super().__init__()
        self.body = MLP(in_channels, out_channels, n_filters, include_head=False)

        self.V = nn.Linear(n_filters[-1], 1).to(device)  # the head for a state-value function
        self.A = nn.Linear(n_filters[-1], out_channels).to(device)  # the head for an advantage-value function

        self.V.register_full_backward_hook(scale_gradients)
        self.A.register_full_backward_hook(scale_gradients)
    
    def forward(self, x):
        s = self.body(x)
        A = self.A(s)
        return self.V(s) + A - A.mean(1).unsqueeze(1)
