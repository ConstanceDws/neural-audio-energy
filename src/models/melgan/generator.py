"""Mainly taken from official code https://github.com/descriptinc/melgan-neurips"""
from .utils import ResnetBlock, WNConv1d, WNConvTranspose1d
from torch.nn import Module, Sequential
from torch.nn import LeakyReLU, ReflectionPad1d, Tanh
# Typing
from torch import Tensor
from typing import List


class Generator(Module):

    # yapf: disable
    def __init__(self,
                 input_size: int,
                 ngf: int,
                 n_residual_layers: int,
                 ratios: List[int]
    ):
        """The generator takes a mel-spectrogram and turns it into a waveform.

        Args:
            `input_size`: Number of mel-spectrogram channels.
            `ngf`: Number of channels.
            `n_residual_layers`: Number of residual blocks.
            `ratios`: Upsampling ratios. If input has shape `[B,C,T]` then the
                    output will be shaped as `[B, C, prod(ratios)]`.
        """
        # yapf: enable
        super().__init__()

        mult = int(2**len(ratios))

        model = [
            ReflectionPad1d(padding=3),
            WNConv1d(in_channels=input_size,
                     out_channels=(mult * ngf),
                     kernel_size=7,
                     padding=0),
        ]

        # Upsample to raw audio scale
        for r in ratios:
            model += [
                LeakyReLU(0.2),
                WNConvTranspose1d(
                    in_channels=(mult * ngf),
                    out_channels=(mult * ngf // 2),
                    kernel_size=(r * 2),
                    stride=r,
                    padding=(r // 2 + r % 2),
                    output_padding=(r % 2),
                ),
            ]

            for j in range(n_residual_layers):
                model += [ResnetBlock(dim=(mult * ngf // 2), dilation=3**j)]

            mult //= 2

        model += [
            LeakyReLU(0.2),
            ReflectionPad1d(padding=3),
            WNConv1d(in_channels=ngf, out_channels=1, kernel_size=7,
                     padding=0),
            Tanh(),
        ]

        self.model = Sequential(*model)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)