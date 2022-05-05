from pathlib import Path
import os
import glob
import numpy as np
import tensorflow as tf


class MyModelCheckpoint(tf.keras.callbacks.Callback):
    """
    Custom checkpoint (saves full model in .h5 format)
    """
    def __init__(self, filepath, monitor='val_loss', save_best_only=True, pattern='model'):
        super(MyModelCheckpoint, self).__init__()

        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.pattern = pattern
        self.best = np.inf

        self.monitor_op = np.less

        if not os.path.exists(filepath):
            os.mkdir(filepath)

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if not self.save_best_only:
            self.model.save(os.path.join(
                self.filepath,
                f"{self.pattern}_{epoch}.h5"))
        elif self.monitor_op(current, self.best):
            msg = f'\nEpoch {epoch + 1}: {self.monitor} improved '
            msg += f'from {self.best:.5f} to {current:.5f}, '
            msg += f'saving model to {self.filepath}'
            print(msg)

            self.best = current
            self.model.save(os.path.join(
                self.filepath,
                f"{self.pattern}_{epoch}.h5"))


class KeepMaxModelCheckpoints(tf.keras.callbacks.Callback):
    """
    Deletes FULL MODEL checkpoints in .h5 format
    if there is more then some threshold

    Models names are in format: {pattern}_{epoch}.h5
    """
    def __init__(self, keep_max, ckpt_dir, pattern='model'):
        super(KeepMaxModelCheckpoints, self).__init__()

        self.keep_max = keep_max
        self.ckpt_dir = ckpt_dir
        self.pattern = pattern

    def on_epoch_end(self, epoch, logs=None):
        files = list(
            glob.glob(os.path.join(self.ckpt_dir, f'{self.pattern}*.h5')))
        files = self._sort_files(files)

        length = len(files)
        to_remove_length = length - self.keep_max

        if to_remove_length > 0:
            files_to_remove = files[:to_remove_length]

            for f in files_to_remove:
                os.remove(f)

    @staticmethod
    def _sort_files(files):
        return sorted(files, key=lambda f: int(Path(f).stem.split('_')[-1]))


class PrintLR(tf.keras.callbacks.Callback):
    def __init__(self, print_every):
        super(PrintLR, self).__init__()

        self.print_every = print_every

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.print_every:
            lr = self.model.optimizer._decayed_lr(tf.float32)
            msg = f"LR at epoch {epoch}: "
            msg += f"{lr.numpy():.6f}"
            print(msg)
