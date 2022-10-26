"""Mainly taken from official code https://github.com/descriptinc/melgan-neurips"""
from turtle import forward
from .generator import Generator
from .discriminator import Discriminator
from pytorch_lightning import LightningModule
from torch.nn.functional import l1_loss, relu
from torch.optim import Adam
import torch
# Typing
from torch import Tensor
from typing import List, Tuple

class MelGAN(LightningModule):
    def __init__(self,
                 n_mel: int,
                 mel_stride: int,
                 ngf: int,
                 n_residual_layers_gen: int,
                 ratios_gen: List[int],
                 num_discriminators: int,
                 ndf: int,
                 n_layers_disc: int,
                 downsamp_factor: int
    ):
        """
        Args:
            `n_mel`: The number of mel channels.
            `mel_stride`: The mel-spectrogram hop-window.
            `ngf`: Number of generator filters (channels).
            `n_residual_layers_gen`: Number of residual layers in the generator.
            `ratios_gen`: List of upsampling factors in the generator.
            `num_discriminators`: Number of different discriminators (acting at different scales).
            `ndf`: Number of discriminator filters (channels).
            `n_layers_disc`: Number of layers in the discriminator.
            `downsamp_factor`: Downsampling factor of each discriminator.
        """
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.name = "melgan"
        self.log_steps_in_an_epoch = 1
        
        # REQUIRED FOR TRAINING
        self.sr = 16000
        self.n_signal = 2**14
        self.n_mel = n_mel
        self.stride = mel_stride

        self.num_discriminators = num_discriminators
        self.wt = (1.0 / num_discriminators) * 4.0 / (n_layers_disc + 1)
        self.lmbda = 10

        self.generator = Generator(input_size=n_mel,
                                   ngf=ngf,
                                   n_residual_layers=n_residual_layers_gen,
                                   ratios=ratios_gen)

        self.discriminator = Discriminator(num_D=num_discriminators,
                                           ndf=ndf,
                                           n_layers=n_layers_disc,
                                           downsampling_factor=downsamp_factor)

    def configure_optimizers(self) -> Tuple[Adam, Adam]:
        g_opt = Adam(
            self.generator.parameters(),
            lr=1e-4,
            betas=(0.5, 0.9),
        )
        d_opt = Adam(
            self.discriminator.parameters(),
            lr=1e-4,
            betas=(0.5, 0.9),
        )
        return g_opt, d_opt

    def training_step(self, batch: Tuple[Tensor, Tensor],
                      batch_idx: int) -> None:
        g_opt, d_opt = self.optimizers()

        x, cdt = batch

        x_hat = self.generator(cdt.detach())
        y_hat = self.discriminator(x_hat.detach())
        y = self.discriminator(x.unsqueeze(1))

        loss_disc = self.discrim_loss(y, y_hat)
        self.discriminator.zero_grad()
        self.manual_backward(loss_disc)
        d_opt.step()

        # Train Generator
        y_hat = self.discriminator(x_hat)

        final_loss_gen = self.gen_loss(y, y_hat)
        self.generator.zero_grad()
        self.manual_backward(final_loss_gen)
        g_opt.step()

        self.log(f"{self.name}/train_loss_gen", final_loss_gen)
        self.log(f"{self.name}/train_loss_disc", loss_disc)
    
    def validation_step(self, batch, batch_idx):
        x, cdt = batch

        x_hat = self.generator(cdt)
        y_hat = self.discriminator(x_hat)
        y = self.discriminator(x.unsqueeze(1))

        disc_loss = self.discrim_loss(y, y_hat)
        gen_loss = self.gen_loss(y, y_hat)

        self.log(f"{self.name}/val_loss", gen_loss)
        self.log(f"{self.name}/val_loss_disc", disc_loss)

    def discrim_loss(self, y: Tensor, y_hat: Tensor) -> Tensor:
        """Computes equation (1) in Sec. 2.3
        See also https://github.com/descriptinc/melgan-neurips/issues/34
        """
        loss_disc = 0
        for scale in y_hat:
            loss_disc += relu(1 + scale[-1]).mean()
        for scale in y:
            loss_disc += relu(1 - scale[-1]).mean()
        return loss_disc

    def gen_loss(self, y: Tensor, y_hat: Tensor) -> Tensor:
        """Computes equation (4) in Sec. 2.3"""
        loss_gen = 0
        for scale in y_hat:
            loss_gen += -scale[-1].mean()
        loss_feat = 0
        for i in range(self.num_discriminators):
            for j in range(len(y_hat[i]) - 1):
                loss_feat += self.wt * l1_loss(y_hat[i][j], y[i][j].detach())
        return loss_gen + self.lmbda * loss_feat

    @torch.no_grad()
    def generate(self, cdt):
        return self.generator(cdt).squeeze(1)

    def forward(self,cdt):
        return self.generate(cdt)
