"""Mainly taken from official code https://github.com/descriptinc/melgan-neurips"""
from torch.nn import Module, Sequential
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn import LeakyReLU, ReflectionPad1d
from torch.nn.utils import weight_norm
# Typing
from torch import Tensor


class WNConv1d(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = weight_norm(Conv1d(*args, **kwargs))
        self.conv.weight.data.normal_(0.0, 0.02)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class WNConvTranspose1d(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = weight_norm(ConvTranspose1d(*args, **kwargs))
        self.conv.weight.data.normal_(0.0, 0.02)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class ResnetBlock(Module):
    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()

        self.block = Sequential(
            LeakyReLU(0.2),
            ReflectionPad1d(dilation),
            WNConv1d(in_channels=dim,
                     out_channels=dim,
                     kernel_size=3,
                     dilation=dilation),
            LeakyReLU(0.2),
            WNConv1d(in_channels=dim, out_channels=dim, kernel_size=1),
        )

        self.shortcut = WNConv1d(in_channels=dim,
                                 out_channels=dim,
                                 kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.shortcut(x) + self.block(x)
