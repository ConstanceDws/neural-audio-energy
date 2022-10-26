import librosa as li
import numpy as np
from einops import rearrange


def get_audio_preprocess(sampling_rate, N, n_mel, stride):
    def preprocess(name):
        try:
            x, sr = li.load(name, sr=sampling_rate)
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(e)
            return None

        # pad = (N - (len(x) % N)) % N
        # x = np.pad(x, (0, pad))
        x = x[:N]

        S = li.feature.melspectrogram(x, sr, hop_length=stride, n_mels=n_mel)
        S = S[..., :len(x) // stride]
        S = np.log(S + 1)

        x = x.reshape(-1, N)
        S = rearrange(S, "f (b t) -> b f t", b=x.shape[0]).astype(np.float32)

        return zip(x.astype(np.float32), S)

    return preprocess