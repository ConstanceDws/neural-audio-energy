import torch
import torch.nn as nn
import numpy as np
from .diffusion import DiffusionModel
from .base_layers import OrthogonalConv1d, UBlock, DBlock, FeaturewiseLinearModulation as FiLM
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class WaveGrad(DiffusionModel):
    def __init__(self, up_dims: list, up_cycle_size: list, up_ratio: list,
                 down_dims: list, down_cycle_size: list, down_ratio: list):
        super().__init__()
        self.ratio = np.prod(up_ratio)
        self.first_conv = OrthogonalConv1d(
            up_dims[0],
            up_dims[1],
            3,
            padding=1,
        )

        ublock_channels = up_dims[1:-1]
        self.upnet = nn.ModuleList([])

        for i in range(len(up_ratio)):
            self.upnet.append(
                UBlock(
                    ublock_channels[i],
                    ublock_channels[i + 1],
                    up_ratio[i],
                    2**(np.arange(4) % up_cycle_size[i]),
                ))

        self.last_conv = OrthogonalConv1d(
            up_dims[-2],
            up_dims[-1],
            3,
            padding=1,
        )

        dblock_channels = down_dims[1:]
        self.dnet = nn.ModuleList(
            [OrthogonalConv1d(down_dims[0], down_dims[1], 5, padding=2)])

        for i in range(len(down_ratio)):
            self.dnet.append(
                DBlock(
                    dblock_channels[i],
                    dblock_channels[i + 1],
                    down_ratio[i],
                    2**(np.arange(3) % down_cycle_size[i]),
                ))

        self.films = nn.ModuleList([])

        for in_size, out_size in zip(dblock_channels, up_dims[2:-1][::-1]):
            self.films.append(FiLM(in_size, out_size))

    def forward(self, y, cdt, noise_level):
        films = []
        for dblock, film in zip(self.dnet, self.films):
            y = dblock(y)
            films.append(film(y, noise_level))

        cdt = self.first_conv(cdt)

        for ublock, (scale, shift) in zip(self.upnet, films[::-1]):
            cdt = ublock(cdt, scale, shift)

        cdt = self.last_conv(cdt)

        return cdt

    def neural_pass(self, y, cdt, noise_level):
        return self.forward(y, cdt, noise_level)

    @property
    def compression_ratio(self):
        return self.ratio

    def do_epoch(self, loader: torch.utils.data.DataLoader,
                 optimizer: torch.optim.Optimizer, writer: SummaryWriter,
                 epoch: int):
        step = len(loader) * epoch
        n_element = 0
        mean_loss = 0
        device = next(iter(self.parameters())).device

        for audio, cdt in tqdm(loader):
            audio = audio.to(device)
            cdt = cdt.to(device).squeeze(1)
            loss = self.compute_loss(audio, cdt)

            if self.training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                writer.add_scalar("loss", loss.item(), step)

                if not step % 1000:
                    gen = self.sample(cdt)[-1]
                    writer.add_audio(
                        "generation",
                        gen.reshape(-1),
                        step,
                        24000,
                    )

            n_element += 1
            mean_loss += (loss.item() - mean_loss) / n_element
            step += 1

        return mean_loss
