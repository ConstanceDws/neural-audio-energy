import torch
import torch.nn as nn
import numpy as np


class OrthogonalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.orthogonal_(self.weight.data)


class Modulation(nn.Module):
    def forward(self, x, scale, shift):
        return scale * x + shift


class Interpolate(nn.Module):
    def __init__(self, ratio, down=False):
        super().__init__()
        self.ratio = ratio
        self.down = down

    def forward(self, x):
        if self.down:
            out_size = x.shape[-1] // self.ratio
        else:
            out_size = x.shape[-1] * self.ratio

        return nn.functional.interpolate(
            x,
            size=out_size,
            mode="linear",
            align_corners=False,
        )


class UBlock(nn.Module):
    def __init__(self, in_size, out_size, ratio, dilations):
        super().__init__()
        dilations = np.asarray(dilations).astype(int)
        full_kernels = 2 * dilations + 1
        paddings = full_kernels // 2

        self.block1 = nn.ModuleList([
            nn.LeakyReLU(),
            Interpolate(ratio),
            OrthogonalConv1d(
                in_size,
                out_size,
                3,
                padding=paddings[0],
                dilation=dilations[0],
            ),
            Modulation(),
            nn.LeakyReLU(),
            OrthogonalConv1d(
                out_size,
                out_size,
                3,
                padding=paddings[1],
                dilation=dilations[1],
            )
        ])

        self.block2 = nn.Sequential(*[
            Interpolate(ratio),
            nn.Conv1d(in_size, out_size, 1),
        ])

        self.block3 = nn.ModuleList([
            Modulation(),
            nn.LeakyReLU(),
            OrthogonalConv1d(
                out_size,
                out_size,
                3,
                padding=paddings[2],
                dilation=dilations[2],
            ),
            Modulation(),
            nn.LeakyReLU(),
            OrthogonalConv1d(
                out_size,
                out_size,
                3,
                padding=paddings[3],
                dilation=dilations[3],
            )
        ])

    def forward(self, x, scale, shift):
        y1 = x.clone()
        for layer in self.block1:
            if isinstance(layer, Modulation):
                y1 = layer(y1, scale, shift)
            else:
                y1 = layer(y1)
        y2 = self.block2(x)
        y = y1 + y2
        y3 = y.clone()
        for layer in self.block3:
            if isinstance(layer, Modulation):
                y3 = layer(y3, scale, shift)
            else:
                y3 = layer(y3)
        return y3 + y


class DBlock(nn.Module):
    def __init__(self, in_size, out_size, ratio, dilations):
        super().__init__()
        dilations = np.asarray(dilations)
        full_kernels = 2 * dilations + 1
        paddings = full_kernels // 2

        self.block1 = nn.Sequential(*[
            Interpolate(ratio, True),
            nn.LeakyReLU(),
            OrthogonalConv1d(
                in_size,
                out_size,
                3,
                padding=paddings[0],
                dilation=dilations[0],
            ),
            nn.LeakyReLU(),
            OrthogonalConv1d(
                out_size,
                out_size,
                3,
                padding=paddings[1],
                dilation=dilations[1],
            ),
            nn.LeakyReLU(),
            OrthogonalConv1d(
                out_size,
                out_size,
                3,
                padding=paddings[2],
                dilation=dilations[2],
            )
        ])

        self.block2 = nn.Sequential(*[
            Interpolate(ratio, True),
            OrthogonalConv1d(in_size, out_size, 1),
        ])

    def forward(self, x):
        return self.block1(x) + self.block2(x)


class PositionalEncoding(nn.Module):
    def __init__(self, n_dim):
        super().__init__()
        self.n_dim = n_dim
        exponents = 1e-4**torch.linspace(0, 1, n_dim // 2)
        self.register_buffer("exponents", exponents)

    def forward(self, noise_level):
        noise_level = noise_level.reshape(-1, 1)
        exponents = self.exponents.unsqueeze(0) * 5000
        encoding = exponents * noise_level
        encoding = torch.cat([encoding.sin(), encoding.cos()], -1)
        return encoding.unsqueeze(-1)


class FeaturewiseLinearModulation(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.convs = nn.ModuleList([
            OrthogonalConv1d(in_size, out_size, 3, padding=1),
            OrthogonalConv1d(out_size, out_size, 3, padding=1),
            OrthogonalConv1d(out_size, out_size, 3, padding=1),
        ])

        self.positional_encoding = PositionalEncoding(out_size)

    def forward(self, x, noise_level):
        x = nn.functional.leaky_relu(self.convs[0](x))
        x = x + self.positional_encoding(noise_level)
        scale = self.convs[1](x)
        shift = self.convs[2](x)
        return scale, shift
