import torch
from .hifigan_repo import models, meldataset
import pytorch_lightning as pl
import numpy as np
from types import SimpleNamespace
import itertools
import torch.nn.functional as F


class HifiGAN(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        config = SimpleNamespace(**kwargs)

        self.generator = models.Generator(config)
        self.msd = models.MultiScaleDiscriminator()
        self.mpd = models.MultiPeriodDiscriminator()
        self.name = "hifigan"

        # REQUIRED FOR TRAINING
        self.sr = 16000
        self.n_signal = 16384  # config.segment_size  # number of audio samples in a batch
        self.n_mel = config.num_mels
        self.stride = np.prod(config.upsample_rates)  # = hop length
        self.config = config
        self.automatic_optimization = False
        self.n_fft = 1024
        
    def configure_optimizers(self):
        optimizer_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.config.learning_rate,
            betas=[self.config.adam_b1, self.config.adam_b2],
        )
        optimizer_d = torch.optim.AdamW(
            itertools.chain(
                self.msd.parameters(),
                self.mpd.parameters(),
            ),
            lr=self.config.learning_rate,
            betas=[self.config.adam_b1, self.config.adam_b2],
        )
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        return optimizer_g, optimizer_d
        # return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        '''
        Batch is composed of a waveform (x) and a mel spectrogram (cdt). 
        '''
        g_opt, d_opt = self.optimizers()
        x, cdt = batch
        x = x.unsqueeze(1)

        y_g_hat = self.generator(cdt)
        y_g_hat_mel = meldataset.mel_spectrogram(y_g_hat.squeeze(1),
                                                 self.n_fft, self.n_mel,
                                                 self.sr,
                                                 self.config.hop_size, self.config.win_size,
                                                 self.config.fmin, self.config.fmax_for_loss)
        x_mel = meldataset.mel_spectrogram(x.squeeze(1), self.n_fft,
                                           self.n_mel, self.sr,
                                           self.config.hop_size, self.config.win_size,
                                           self.config.fmin, self.config.fmax_for_loss)

        d_opt.zero_grad()

        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(x, y_g_hat.detach())
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = models.discriminator_loss(
            y_df_hat_r, y_df_hat_g)

        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(x, y_g_hat.detach())
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = models.discriminator_loss(
            y_ds_hat_r, y_ds_hat_g)

        loss_disc_all = loss_disc_s + loss_disc_f

        loss_disc_all.backward()
        d_opt.step()

        # Generator
        g_opt.zero_grad()

        # L1 Mel-Spectrogram Loss
        loss_mel = F.l1_loss(x_mel, y_g_hat_mel) * 45

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(x, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(x, y_g_hat)
        loss_fm_f = models.feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = models.feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = models.generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = models.generator_loss(y_ds_hat_g)
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

        loss_gen_all.backward()
        g_opt.step()

        self.log(f"{self.name}/gen_loss", loss_gen_all)
        self.log(f"{self.name}/dis_loss", loss_disc_all)

    def validation_step(self, batch, batch_idx):
        '''
        Batch is composed of a waveform (x) and a mel spectrogram (cdt). 
        '''
        x, cdt = batch
        x = x.unsqueeze(1)

        y_g_hat = self.generator(cdt)
        y_g_hat_mel = meldataset.mel_spectrogram(y_g_hat.squeeze(1),
                                                 self.n_fft, self.n_mel,
                                                 self.sr,
                                                 self.config.hop_size, self.config.win_size,
                                                 self.config.fmin, self.config.fmax_for_loss)
        x_mel = meldataset.mel_spectrogram(x.squeeze(1), self.n_fft,
                                           self.n_mel, self.sr,
                                           self.config.hop_size, self.config.win_size,
                                           self.config.fmin, self.config.fmax_for_loss)

        # L1 Mel-Spectrogram Loss
        loss_mel = F.l1_loss(x_mel, y_g_hat_mel) * 45

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(x, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(x, y_g_hat)
        loss_fm_f = models.feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = models.feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = models.generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = models.generator_loss(y_ds_hat_g)
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

        self.log(f"{self.name}/validation", loss_gen_all)

    @torch.no_grad()
    def generate(self, cdt):
        '''
        takes a single tensor of shape B X N_MEL X T
        returns an audio waveform of shape B x T
        '''
        return self.generator(cdt).squeeze(1)

    def forward(self, cdt):
        return self.generate(cdt)