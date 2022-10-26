from .dummy.dummy import Dummy
from .wavenet.model import Wavenet
from .wavegrad import WaveGrad
from .melgan.model import MelGAN
from .waveflow.model import WaveFlow
from .hifigan import HifiGAN
from .diffwave.model import DiffWave
from .waveglow.model import WaveGlow

import yaml
from glob import glob
import os


def path_to_dict(path):
    name = os.path.splitext(os.path.basename(path))[0]
    with open(path, "r") as path:
        path = yaml.safe_load(path)
    return name, path


def get_variant_dict(model_name):
    path = os.path.join("energetic_training", "models", model_name, "*.yaml")
    variant = map(path_to_dict, glob(path))
    return {k: v for k, v in variant}


# ADD NEW MODELS HERE !
available_models = {
    # "dummy": Dummy,
    "wavenet": Wavenet,
    "wavegrad": WaveGrad,
    "melgan": MelGAN,
    "waveflow": WaveFlow,
    "hifigan": HifiGAN,
    "diffwave" : DiffWave,
    "waveglow": WaveGlow,
}
