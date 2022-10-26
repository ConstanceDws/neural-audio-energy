"""Taken from official code https://github.com/descriptinc/melgan-neurips"""
from .utils import WNConv1d
from torch.nn import Module, ModuleDict, Sequential
from torch.nn import AvgPool1d, LeakyReLU, ReflectionPad1d
# Typing
from torch import Tensor
from typing import List


class Discriminator(Module):

    # yapf: disable
    def __init__(self,
                 num_D: int,
                 ndf: int,
                 n_layers: int,
                 downsampling_factor: int
    ):
        """Args:
            `num_D`: Number of different discriminators (acting at different scales).
            `ndf`: Number of discriminator filters (channels).
            `n_layers`: Number of layers in the discriminator.
            `downsampling_factor`: Downsampling factor of each discriminator.
        """
        # yapf: enable
        super().__init__()

        self.model = ModuleDict()

        for i in range(num_D):
            self.model[f"disc_{i}"] = NLayerDiscriminator(
                ndf=ndf,
                n_layers=n_layers,
                downsampling_factor=downsampling_factor)

        self.downsample = AvgPool1d(kernel_size=4,
                                    stride=2,
                                    padding=1,
                                    count_include_pad=False)

    def forward(self, x: Tensor) -> List[List[Tensor]]:
        results = []
        for _, disc in self.model.items():
            results.append(disc(x))
            x = self.downsample(x)
        return results


class NLayerDiscriminator(Module):
    def __init__(self, ndf: int, n_layers: int, downsampling_factor: int):
        super().__init__()

        model = ModuleDict()

        model["layer_0"] = Sequential(
            ReflectionPad1d(padding=7),
            WNConv1d(in_channels=1, out_channels=ndf, kernel_size=15),
            LeakyReLU(0.2, True),
        )

        nf = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):

            nf_prev = nf
            nf = min(nf * stride, 1024)

            model[f"layer_{n}"] = Sequential(
                WNConv1d(
                    in_channels=nf_prev,
                    out_channels=nf,
                    kernel_size=(stride * 10 + 1),
                    stride=stride,
                    padding=(stride * 5),
                    groups=(nf_prev // 4),
                ),
                LeakyReLU(0.2, True),
            )

        nf = min(nf * 2, 1024)
        model[f"layer_{n_layers + 1}"] = Sequential(
            WNConv1d(in_channels=nf_prev,
                     out_channels=nf,
                     kernel_size=5,
                     stride=1,
                     padding=2),
            LeakyReLU(0.2, True),
        )

        model[f"layer_{n_layers + 2}"] = WNConv1d(in_channels=nf,
                                                  out_channels=1,
                                                  kernel_size=3,
                                                  stride=1,
                                                  padding=1)

        self.model = model

    def forward(self, x: Tensor) -> List[Tensor]:
        results = []
        for _, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results
