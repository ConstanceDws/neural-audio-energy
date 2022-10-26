import torch
from typing import List


def spectral_distance(x: torch.Tensor,
                      y: torch.Tensor,
                      multiband: bool = True):
    if multiband:
        return multiband_spectral_dist(x, y)
    else:
        return spectral_dist(x, y)

def spectral_dist(x: torch.Tensor,
                  y: torch.Tensor,
                  n_fft: int = 1024,
                  overlap: float = 0.75,
                  eps: float = 1.):
    sx, sy = power_s(x, n_fft, overlap), power_s(y, n_fft, overlap)
    return abs(torch.log(eps + sx) - torch.log(eps + sy)).mean()

def multiband_spectral_dist(x: torch.Tensor,
                            y: torch.Tensor,
                            scales: List[int] = [2048, 1024, 512, 256],
                            overlap: float = 0.75,
                            eps: float = 1.):
    return sum([spectral_dist(x, y, f, overlap, eps) for f in scales])

def power_s(x: torch.Tensor,
            n_fft: int,
            overlap: float):
    s = torch.stft(input=x,
                   n_fft=n_fft,
                   hop_length=int(n_fft * (1 - overlap)),
                   win_length=n_fft,
                   window=torch.hann_window(n_fft).to(x),
                )
    power_s = s[..., 0]**2 + s[..., 1]**2
    return power_s
