import speechmetrics as sm
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import tensorflow as tf
import argparse
import os

from src import PeriodicCheckpoint, accumulate_metric_dict
from src.models import available_models, get_variant_dict
from src.datasets import datasets, get_dataset
from src.utils import spectral_distance
from deepspeed.profiling.flops_profiler import get_model_profile



parser = argparse.ArgumentParser()

parser.add_argument(
    "model",
    metavar="model",
    type=str,
    help=f"name of the model to train ({' | '.join(available_models.keys())})",
)
parser.add_argument(
    "--variant",
    type=str,
    default="default",
    help="variant of the model to use",
)
parser.add_argument(
    "--dataset",
    type=str,
    default='ljspeech',
    help=f"name of the dataset ({' | '.join(datasets.keys())})",
)
parser.add_argument(
    "--ckpt",
    type=str,
    default=None,
    help=f"checkpoint to resume",
)
parser.add_argument(
    '--no-carbon',
    action="store_true",
    default=False,
)
parser.add_argument(
    "--gpu",
    type=int,
    default=0,
)
parser.add_argument("--epochs",
                    type=int,
                    default=10000,
                    help="number of epochs")
parser.add_argument("--output",
                    type=str,
                    default='runs',
                    help="output directory logs")
parser.add_argument(
    "--fix_split",
    action="store_true",
    default=True,
)
parser.add_argument("--gradient_clipping",
                    type=int,
                    default=None,
                    help="Gradient clipping")

parser.add_argument("--batch_size", type=int, default=16, help="batch size")

parser.add_argument("--no-scale", action="store_true", default=False)

args = parser.parse_args()

#Sets only the GPU used for training
if args.gpu >= 0:
    gpu_or_cpu = [args.gpu]

    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(physical_devices[args.gpu], "GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    torch.cuda.set_device(
        args.gpu)  # Needed for waveglow, o/w raises error on device TODO fix
else:
    gpu_or_cpu = None
print("GPU : ", gpu_or_cpu)


class Model(available_models[args.model]):

    def __init__(self) -> None:
        super().__init__(**get_variant_dict(args.model)[args.variant])

        self.batch_size = args.batch_size
        self.train_set = None
        self.val_set = None
        self.num_workers = 0
        self.name = args.model

    def set_dataset(self, train, val):
        self.train_set = train
        self.val_set = val

        self.len_train_loader = len(self.train_set) // self.batch_size + 1
        self.len_val_loader = len(self.val_set) // self.batch_size + 1

    def on_fit_start(self):
        tb = self.logger.experiment
        tb.add_scalar(f"{self.name}/batch_size", self.batch_size)
        tb.add_scalar(f"{self.name}/n_batches_train", self.len_train_loader)
        tb.add_scalar(f"{self.name}/n_batches_val", self.len_val_loader)
        return

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    @torch.no_grad()
    def validation_epoch_end(self, *args, **kwargs):
        loader = self.val_dataloader()
        tb = self.logger.experiment
        oris = []
        recs = []
        num_valid = 1
        counter = 0
        for audio, melspec in loader:
            while counter < num_valid:
                audio = audio.cuda()
                melspec = melspec.cuda()

                reconstruction = self.generate(melspec)
                oris.append(audio)
                recs.append(reconstruction)
                counter = counter + 1

        oris = torch.cat(oris, 0)
        recs = torch.cat(recs, 0)
        # COMPUTE METRICS
        spec_dist = spectral_distance(oris, recs, False)
        tb.add_scalar(f"{self.name}/valid_spectral_dist", spec_dist,
                      self.global_step)

        acc_metrics = None
        metrics = sm.load(["mosnet", "srmr", "bsseval",
                           "nb_pesq"])  #,"sisdr","pesq"])
        for i, (o, r) in enumerate(zip(oris, recs)):
            o = o.cpu().numpy().reshape(-1)
            r = r.cpu().numpy().reshape(-1)
            new_metrics = metrics(r, o, rate=self.sr)
            acc_metrics = accumulate_metric_dict(acc_metrics, new_metrics, i)

        for k, v in acc_metrics.items():
            tb.add_scalar(f"{self.name}/{k}", v, self.global_step)

        # LOG AUDIO
        audio = torch.cat([oris[:32], recs[:32]], 1).flatten()
        tb.add_audio(f"{self.name}", audio, self.global_step, self.sr)


m = Model()

m.set_dataset(*get_dataset(
    args.dataset,
    m.sr,
    m.n_signal,
    m.n_mel,
    m.stride,
    deterministic=args.fix_split,
))

#Create tensorboard loader
tb_logger = pl.loggers.TensorBoardLogger(os.path.join(args.output, args.model,
                                                      args.variant),
                                         name=args.dataset,
                                         default_hp_metric=False)

tb_logger_dir = tb_logger.log_dir

trainer = pl.Trainer(
    logger=tb_logger,
    gpus=gpu_or_cpu,
    max_epochs=args.epochs,
    max_time="05:00:00:00",
    auto_scale_batch_size=not args.no_scale,
    callbacks=[
        PeriodicCheckpoint(
            logdir=tb_logger_dir,
            use_tracker=not args.
            no_carbon,  #Monitor energy and carbon consumption
            every_n_epochs=10,  #Save checkpoints 
            save_top_k=-1,  #Do not overwright the checkpoints
        )
    ],
    gradient_clip_val=args.gradient_clipping,
)

trainer.tune(m)
print("Training starts")
trainer.fit(m, ckpt_path=args.ckpt)
print("Training stops")
