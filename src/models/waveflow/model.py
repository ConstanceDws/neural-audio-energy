from math import log, pi
from pytorch_lightning import LightningModule
from torch.optim import Adam
from .flow import WaveFlowModel
import torch
# Typing
from torch import Tensor
from typing import Tuple

class WaveFlow(LightningModule):

    def __init__(self,
                 sr: int,
                 n_signal: int,
                 n_mel: int,
                 mel_stride: int,
                 in_channels: int,
                 res_channels: int,
                 n_height: int,
                 n_flow: int,
                 n_layer: int,
                 layers_per_dilation_h_cycle: int,
                 bipartize: bool,
                 sigma: float):
        super().__init__()
        self.save_hyperparameters()
        self.name = "waveflow"
        self.automatic_optimization = False
        self.log_steps_in_an_epoch = 3

        # REQUIRED FOR TRAINING
        self.sr = sr
        self.n_signal = n_signal
        self.n_mel = n_mel
        self.stride = mel_stride

        self.model = WaveFlowModel(
            in_channel=in_channels,
            cin_channel=n_mel,
            res_channel=res_channels,
            n_height=n_height,
            n_flow=n_flow,
            n_layer=n_layer,
            layers_per_dilation_h_cycle=layers_per_dilation_h_cycle,
            upscaling_fact=(mel_stride // 16),
            bipartize=bipartize)

        self.sigma = sigma
        self._sigmasq = sigma**2
        self._z_coeff = 1. / (2 * sigma**2)
        self._const = 0.5 * log(2 * pi) + log(sigma)

    def loss(self, out: Tensor, logdet: Tensor) -> Tensor:
        logdet = logdet.sum()
        B, _, C, T = out.size()
        loss = (0.5) * (log(2.0 * pi) + 2 * log(self.sigma) + out.pow(2) /
                        (self.sigma * self.sigma)).sum() - logdet
        return loss / (B * C * T)

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=2e-4)

    def training_step(self, batch: Tuple[Tensor, Tensor],
                      batch_idx: int) -> Tensor:
        opt = self.optimizers()
        x, cdt = batch
        out, logdet = self.model(x, cdt)
        loss = self.loss(out, logdet)
        self.model.zero_grad()
        self.manual_backward(loss)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
        opt.step()
        # Logging
        self.log(f"{self.name}/train_loss", loss)
        # self.log("waveflow/z", {"mean": out.mean(), "std": out.std()})
        self.log(f"{self.name}/logdet-avg", logdet.mean())
        self.log(f"{self.name}/grad_norm", grad_norm)

        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor],
                        batch_idx: int) -> None:
        x, cdt = batch
        out, logdet = self.model(x, cdt)
        loss = self.loss(out, logdet)
        # Logging
        self.log(f"{self.name}/val_loss", loss)
    
    @torch.no_grad()
    def generate(self, cdt):
        return  self.model.reverse_fast(cdt)

    def forward(self,cdt):
        return self.generate(cdt)