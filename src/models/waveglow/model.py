from .waveglow import WaveGlow as WaveGlowModel
from pytorch_lightning import LightningModule
from torch.optim import Adam
import torch
# Typing
from torch import Tensor
from typing import Tuple


class WaveGlow(LightningModule):

    #yapf: disable
    def __init__(self,
                 sr: int,
                 n_signal: int,
                 n_mel: int,
                 mel_stride: int,
                 n_flows: int,
                 n_group: int,
                 n_early_every: int,
                 n_early_size: int,
                 n_layers_wn: int,
                 n_channels_wn: int,
                 kernel_size_wn: int,
                 sigma: float):
        super().__init__()
        self.save_hyperparameters()
        self.name = "waveglow"
        self.automatic_optimization = False
        self.log_steps_in_an_epoch = 3

        # REQUIRED FOR TRAINING
        self.sr = sr
        self.n_signal = n_signal
        self.n_mel = n_mel
        self.stride = mel_stride

        self.sigma = sigma

        self.model = WaveGlowModel(n_mel, n_flows, n_group, n_early_every,
                 n_early_size, n_layers_wn, n_channels_wn, kernel_size_wn)

    def loss(self, z, log_s_list, log_det_W_list) -> Tensor:
        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s)
                log_det_W_total = log_det_W_list[i]
            else:
                log_s_total = log_s_total + torch.sum(log_s)
                log_det_W_total += log_det_W_list[i]

        loss = torch.sum(z * z) / (2 * self.sigma *
                                   self.sigma) - log_s_total - log_det_W_total
        return loss / (z.size(0) * z.size(1) * z.size(2))

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=1e-4)

    def training_step(self, batch: Tuple[Tensor, Tensor],
                      batch_idx: int) -> Tensor:
        opt = self.optimizers()
        x, cdt = batch
        out, logdet_s, logdet_w = self.model(x, cdt)
        loss = self.loss(out, logdet_s, logdet_w)
        self.model.zero_grad()
        self.manual_backward(loss)
        opt.step()
        # Logging
        self.log(f"{self.name}/train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor],
                        batch_idx: int) -> None:
        x, cdt = batch
        out, logdet_s, logdet_w = self.model(x, cdt)
        loss = self.loss(out, logdet_s, logdet_w)
        # Logging
        self.log(f"{self.name}/val_loss", loss)
    
    @torch.no_grad()
    def generate(self, cdt):
        return self.model.infer(cdt)

    def forward(self,cdt):
        return self.generate(cdt)
