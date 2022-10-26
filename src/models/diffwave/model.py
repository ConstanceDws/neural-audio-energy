import torch
import torch.nn as nn
import pytorch_lightning as pl


import numpy as np
from .diffwave import DiffWaveBase

class DiffWave(pl.LightningModule):
    def __init__(self, residual_layers, residual_channels,dilation_cycle_length) -> None:
        super().__init__()
        self.loss_fn = nn.L1Loss()
        self.name = "diffwave"
        # REQUIRED FOR TRAINING
        self.sr = 16000
        self.n_signal = 2**14  # number of audio samples in a batch
        self.n_mel = 80
        self.stride = 256  # = hop length
        self.crop_mel_frames: 62  # Probably an error in paper.
        self.noise_schedule = np.linspace(1e-4, 0.05, 50).tolist()
        self.inference_schedule = [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5]
        self.model = DiffWaveBase(self.n_mel, residual_layers, residual_channels, dilation_cycle_length, self.noise_schedule)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=2e-4)

    def training_step(self, batch):
        x, cdt = batch
        N, T = x.shape
        
        beta = np.array(self.noise_schedule)
        noise_level = np.cumprod(1 - beta)
        noise_level = torch.tensor(noise_level.astype(np.float32), device = cdt.device)

        t = torch.randint(0, len(self.noise_schedule), [N], device = cdt.device)
        noise_scale = noise_level[t].unsqueeze(1)
        noise_scale_sqrt = noise_scale**0.5
        noise = torch.randn_like(x)
        noisy_audio = noise_scale_sqrt * x + (1.0 - noise_scale)**0.5 * noise

        predicted = self.model(noisy_audio, t, cdt)
        loss = self.loss_fn(noise, predicted.squeeze(1))
        
        self.log(f"{self.name}/train_loss", loss)

        return loss


    def validation_step(self, batch, batch_idx):
        x, cdt = batch
        N, T = x.shape
        
        beta = np.array(self.noise_schedule)
        noise_level = np.cumprod(1 - beta)
        noise_level = torch.tensor(noise_level.astype(np.float32), device = cdt.device)

        t = torch.randint(0, len(self.noise_schedule), [N], device = cdt.device)
        noise_scale = noise_level[t].unsqueeze(1)
        noise_scale_sqrt = noise_scale**0.5
        noise = torch.randn_like(x)
        noisy_audio = noise_scale_sqrt * x + (1.0 - noise_scale)**0.5 * noise

        predicted = self.model(noisy_audio, t, cdt)
        loss = self.loss_fn(noise, predicted.squeeze(1))
        
        self.log(f"{self.name}/val_loss", loss)
        pass

    
    @torch.no_grad()
    def generate(self,cdt):
        training_noise_schedule = np.array(self.noise_schedule)
        inference_noise_schedule = np.array(self.inference_schedule)


        talpha = 1 - training_noise_schedule
        talpha_cum = np.cumprod(talpha)

        beta = inference_noise_schedule
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)

        T = []
        for s in range(len(inference_noise_schedule)):
            for t in range(len(training_noise_schedule) - 1):
                if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
                    twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
                    T.append(t + twiddle)
                    break
        T = np.array(T, dtype=np.float32)
        
        if len(cdt.shape) == 2:# Expand rank 2 tensors by adding a batch dimension.
            cdt = cdt.unsqueeze(0)
        audio = torch.randn(cdt.shape[0], self.stride * cdt.shape[-1], device = cdt.device)

        for n in range(len(alpha) - 1, -1, -1):
            c1 = 1 / alpha[n]**0.5
            c2 = beta[n] / (1 - alpha_cum[n])**0.5
            audio = c1 * (audio - c2 * self.model(audio, torch.tensor([T[n]], device = cdt.device), cdt).squeeze(1))
        if n > 0:
            noise = torch.randn_like(audio)
            sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
            audio += sigma * noise
        audio = torch.clamp(audio, -1.0, 1.0)

        return audio

    def forward(self,cdt):
        return self.generate(cdt)