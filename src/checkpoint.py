from pytorch_lightning.callbacks import ModelCheckpoint
from carbontracker.tracker import CarbonTracker
from sys import path


class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, logdir, use_tracker, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.logdir = logdir
        self.tracker = None
        self.use_tracker = use_tracker

    def on_train_epoch_start(self, trainer, pl_module):
        if self.tracker is None and self.use_tracker:
            self.tracker = CarbonTracker(
                epochs =trainer.max_epochs,  #Total epochs of the training loop.
                monitor_epochs= 1,  #Total number of epochs to monitor. Outputs actual consumption when reached. Set to -1 for all epochs.
                devices_by_pid= True,  #If True, only devices (under the chosen components) running processes associated with the main process are measured. If False, all available devices are measured 
                log_dir= self.logdir + "/carbon",  #Path to the desired directory to write log files
            )
            self.tracker.epoch_start()
        return super().on_train_epoch_start(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.tracker is not None:
            self.tracker.epoch_end()
        return super().on_train_epoch_end(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        if self.tracker is not None:
            self.tracker.stop()
        return super().on_train_end(trainer, pl_module)
