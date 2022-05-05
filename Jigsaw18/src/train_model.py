import os
import re
from tensorflow.keras.callbacks import TensorBoard

from .callbacks import MyModelCheckpoint, KeepMaxModelCheckpoints


def _make_config(config):
    output = ''
    for key, val in config.items():

        if isinstance(val, dict):
            val = _make_config(val)
            indent = '\n' + ' ' * (len(key) + 2)
            val = re.sub('\n', indent, val)
            val = val.rstrip()

        output += f'{key}: {val}\n'

    return output


def _print_config(config) -> None:
    delimiter = '.' * 40
    config = _make_config(config)
    config = delimiter + '\nCONFIG:\n' + config + '\n' + delimiter
    print(config)


def train_model(config: dict, strategy):
    main_dir = config['main_dir']
    experiment_name = config['experiment_name']

    log_dir = os.path.join(main_dir, 'tensorboard_logs', experiment_name)
    checkpoint_dir = os.path.join(main_dir, 'checkpoints', experiment_name)

    os.mkdir(log_dir)
    os.mkdir(checkpoint_dir)

    train_data = config['train_ds_class'](**config['train_ds_config'])
    val_data = config['val_ds_class'](**config['val_ds_config'])

    compile_config = {
        'optimizer': config['optimizer'](**config['optimizer_config']),
        'loss': config['loss'],
        'metrics': config.pop('metrics', [])
    }

    with strategy.scope():
        model = config['model'](**config['model_config'])
        model.compile(**compile_config)

    tensorboard_callback = TensorBoard(log_dir=log_dir)

    checkpoint_config = {
        'filepath': checkpoint_dir,
        'pattern': 'model'
    }

    checkpoint_callback = MyModelCheckpoint(**checkpoint_config)

    keep_max_config = {
        'keep_max': 1,
        'pattern': 'model',
    }

    keep_max_callback = KeepMaxModelCheckpoints(**keep_max_config)

    extra_callbacks = config.pop('extra_callbacks', [])

    callbacks = [tensorboard_callback, checkpoint_callback, keep_max_callback]\
                + extra_callbacks

    epochs = config['epochs']

    _print_config(config)

    training_history = model.fit(
        train_data.dataset,
        validation_data=val_data.dataset,
        epochs=epochs,
        steps_per_epoch=train_data.n_steps,
        validation_steps=val_data.n_steps,
        callbacks=callbacks
    )

    return training_history
