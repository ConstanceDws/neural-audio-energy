# from sched import scheduler
from re import L
import torch
from .wavegrad import WaveGrad as WaveGradBase
import pytorch_lightning as pl
import numpy as np


class WaveGrad(pl.LightningModule):

    def __init__(
        self,
        up_dims,
        up_cycle_size,
        up_ratio,
        down_dims,
        down_cycle_size,
        down_ratio,
    ):
        super().__init__()
        self.net = WaveGradBase(
            up_dims,
            up_cycle_size,
            up_ratio,
            down_dims,
            down_cycle_size,
            down_ratio,
        )
        self.name = "wavegrad"

        # REQUIRED FOR TRAINING
        self.sr = 16000
        self.n_signal = 2**14  # number of audio samples in a batch
        self.n_mel = up_dims[0]
        self.stride = np.prod(up_ratio)  # = hop length

        self.net.set_noise_schedule()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, .9)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        '''
        Batch is composed of a waveform (x) and a mel spectrogram (cdt). 
        '''
        x, cdt = batch
        x = x.unsqueeze(1)

        loss = self.net.compute_loss(x, cdt)
        self.log(f"{self.name}/train_loss", loss)
        # print(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        '''
        Batch is composed of a waveform (x) and a mel spectrogram (cdt). 
        '''
        x, cdt = batch
        x = x.unsqueeze(1)

        loss = self.net.compute_loss(x, cdt)
        self.log(f"{self.name}/val_loss", loss)
        return loss

    @torch.no_grad()
    def generate(self, cdt):
        '''
        takes a single tensor of shape B X N_MEL X T
        returns an audio waveform of shape B x T
        '''
        return self.net.sample(cdt)[-1].squeeze(1)

    def forward(self, cdt):
        return self.generate(cdt)