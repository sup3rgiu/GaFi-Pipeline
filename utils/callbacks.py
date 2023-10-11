import tensorflow as tf
import keras
from utils import misc
import math
import os
import numpy as np
import pandas as pd
from keras.utils import io_utils
from keras.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rc('font', size=16) 

tfk = tf.keras

class LogBestAccuracyCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.current_best_acc = -1.0
    
    def on_epoch_end(self, epoch, logs=None):
        logs = keras.utils.tf_utils.sync_to_numpy_or_python_type(logs)
        current_acc = logs.get('val_accuracy')
        if current_acc > self.current_best_acc:
            self.current_best_acc = current_acc
            
        print("Current best validation accuracy: Top-1: {top1:.2f}%\n".format(top1=self.current_best_acc*100.))


class ResnetCifarLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate):
        self.initial_learning_rate = initial_learning_rate
        self.current_learning_rate = initial_learning_rate

    def __call__(self, step):
        # drop LR by a factor 10 at 32k and 48k iterations
        
        def decay_lr():
            self.current_learning_rate = self.current_learning_rate / 10
            return self.current_learning_rate
        
        def return_current_lr():
            return self.current_learning_rate
        
        condition = tf.logical_or(
            tf.equal(step, 32000),
            tf.equal(step, 48000),
        )
        
        return tf.cond(condition, decay_lr,  return_current_lr)


class AdamLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate, decay_start, total_iterations, optimizer_step):
        self.initial_learning_rate = initial_learning_rate
        self.decay_start = decay_start
        self.total_iterations = total_iterations
        self.optimizer_step = optimizer_step

    def __call__(self, step):
        
        step = step // self.optimizer_step
        decay = tf.maximum(0., 1.-(tf.maximum(0., tf.cast(step, tf.float32) -
                                              self.decay_start))/self.total_iterations)
        
        return self.initial_learning_rate * decay


class WarmupCosineDecayLRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, 
                 max_lr: float,
                 warmup_steps: int,
                 decay_steps: int,
                 alpha: float = 0.) -> None:
        super(WarmupCosineDecayLRScheduler, self).__init__()

        self.name = 'WarmupCosineDecayLRScheduler'
        self.alpha = alpha

        self.max_lr = max_lr
        self.last_step = 0

        self.warmup_steps = int(warmup_steps)
        self.linear_increase = self.max_lr / float(self.warmup_steps)

        self.decay_steps = int(decay_steps)

    def _decay(self) -> tf.Tensor:
        rate = tf.subtract(self.last_step, self.warmup_steps) 
        rate = tf.divide(rate, self.decay_steps)
        rate = tf.cast(rate, tf.float32)

        cosine_decayed = tf.multiply(tf.constant(math.pi), rate)
        cosine_decayed = tf.add(1., tf.cos(cosine_decayed))
        cosine_decayed = tf.multiply(.5, cosine_decayed)

        decayed = tf.subtract(1., self.alpha)
        decayed = tf.multiply(decayed, cosine_decayed)
        decayed = tf.add(decayed, self.alpha)
        return tf.multiply(self.max_lr, decayed)

    def __call__(self, step: int) -> tf.Tensor:
        self.last_step = step
        return tf.cond(
            tf.less(self.last_step, self.warmup_steps),
            lambda: tf.multiply(self.linear_increase, self.last_step),
            lambda: self._decay())

    def get_config(self) -> dict:
        config = {
            "max_lr": self.max_lr,
            "warmup_steps": self.warmup_steps,
            'decay_steps': self.decay_steps,
            'alpha': self.alpha
        }
        return config


class LearningRateStepScheduler(tf.keras.callbacks.LearningRateScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_global_step = 0
    
    def on_epoch_begin(self, epoch, logs=None):
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        pass
    
    def on_train_batch_begin(self, batch, logs=None):
        self.current_global_step += 1
        super().on_epoch_begin(self.current_global_step, logs)
        
    def on_train_batch_end(self, batch, logs=None):
        super().on_epoch_end(batch, logs)


class LearningRateEpochScheduler(tf.keras.callbacks.LearningRateScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_epoch = 0
    
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch += 1
        super().on_epoch_begin(self.current_epoch, logs)


class WeightScheduler(tf.keras.callbacks.Callback):
    def __init__(self, variable_name, schedule_fn):
        self.variable_name = variable_name
        self.schedule_fn = schedule_fn
        
        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.current_epoch += 1
        epoch = self.current_epoch # remove this if this callback is used in a .fit with more than 1 epoch
 
        if hasattr(self.model, self.variable_name):
            new_value = self.schedule_fn(epoch)
            variable = getattr(self.model, self.variable_name)
            tf.keras.backend.set_value(variable, new_value)


class MultiModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """
    Callback to save the Keras model or model weights at some frequency.
    Like tf.keras.callbacks.ModelCheckpoint but supporting multiple metrics to monitor.    
    """

    def __init__(
        self,
        filepath,
        monitor: str = "val_loss",
        verbose: int = 0,
        save_best_only: bool = False,
        save_weights_only: bool = False,
        mode: str = "auto",
        save_freq="epoch",
        options=None,
        initial_value_threshold=None,
        sub_model_name=None,
        include_optimizer=True,
        **kwargs,
    ):
        #super().__init__() # DON'T!
        self._supports_tf_logs = True
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = io_utils.path_to_string(filepath)
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_freq = save_freq
        self.epochs_since_last_save = 0
        self._batches_seen_since_last_saving = 0
        self._last_batch_seen = 0
        self.initial_value_threshold = initial_value_threshold
        self.sub_model_name = sub_model_name
        self.include_optimizer = include_optimizer
        
        self.current_epoch = 0

        if save_weights_only:
            if options is None or isinstance(options, tf.train.CheckpointOptions):
                self._options = options or tf.train.CheckpointOptions()
            else:
                raise TypeError(
                    "If save_weights_only is True, then `options` must be "
                    "either None or a tf.train.CheckpointOptions. "
                    f"Got {options}."
                )
            if include_optimizer == False and self.filepath[-3:] != '.h5':
                raise TypeError(
                    "If you don't want to save optimizer state when using `save_weights_only`, "
                    "you must use HDF5 format and not TF format. Please add `.h5` as extension type "
                    "to your `filepath`"
                )
        else:
            if options is None or isinstance(options, tf.saved_model.SaveOptions):
                self._options = options or tf.saved_model.SaveOptions()
            else:
                raise TypeError(
                    "If save_weights_only is False, then `options` must be "
                    "either None or a tf.saved_model.SaveOptions. "
                    f"Got {options}."
                )
        
        self.load_weights_on_restart = False

        self.period = 1

        if not isinstance(mode, list):
            mode = [mode]
        if not isinstance(self.monitor, list):
            self.monitor = [self.monitor]
        if isinstance(self.initial_value_threshold, list):
            assert len(self.initial_value_threshold) == len(self.monitor), "Number of initial_value_threshold do not match number of monitored metrics."
        assert len(mode) == len(self.monitor), "Number of mode do not match number of monitored metrics."
        
        
        _mode = []
        for m in mode:
            if m not in ["auto", "min", "max"]:
                logging.warning(
                    "ModelCheckpoint mode %s is unknown, fallback to auto mode.",
                    mode,
                )
                m = "auto"
            _mode.append(m)
        mode = _mode

        self.best = []
        self.monitor_op = []
        for i, m in enumerate(mode):
            if m == "min":
                self.monitor_op += [np.less]
                curr_best = np.Inf
                if self.initial_value_threshold is not None:
                    curr_best = self.initial_value_threshold[i]
                self.best += [curr_best]
            elif m == "max":
                self.monitor_op += [np.greater]
                curr_best = -np.Inf
                if self.initial_value_threshold is not None:
                    curr_best = self.initial_value_threshold[i]
                self.best += [curr_best]
            else:
                if "acc" in self.monitor or self.monitor.startswith("fmeasure"):
                    self.monitor_op += [np.greater]
                    curr_best = -np.Inf
                    if self.initial_value_threshold is not None:
                        curr_best = self.initial_value_threshold[i]
                    self.best += [curr_best]
                else:
                    self.monitor_op += [np.less]
                    curr_best = np.Inf
                    if self.initial_value_threshold is not None:
                        curr_best = self.initial_value_threshold[i]
                    self.best += [curr_best]

        if self.save_freq != "epoch" and not isinstance(self.save_freq, int):
            raise ValueError(
                f"Unrecognized save_freq: {self.save_freq}. "
                'Expected save_freq are "epoch" or integer'
            )

        # Only the chief worker writes model checkpoints, but all workers
        # restore checkpoint at on_train_begin().
        self._chief_worker_only = False
        
    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch=self.current_epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch=self.current_epoch, logs=logs)
        self.current_epoch += 1

    def _save_model(self, epoch, batch, logs):
        """Saves the model.
        Args:
            epoch: the epoch this iteration is in.
            batch: the batch this iteration is in. `None` if the `save_freq`
              is set to `epoch`.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}

        if (
            isinstance(self.save_freq, int)
            or self.epochs_since_last_save >= self.period
        ):
            # Block only when saving interval is reached.
            logs = tf_utils.sync_to_numpy_or_python_type(logs)
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, batch, logs)

            try:
                if self.save_best_only:
                    improved_metrics = []
                    for i, (metric, monitor_op, best) in enumerate(zip(self.monitor, self.monitor_op, self.best)):
                        current = logs.get(metric)
                        if current is None:
                            logging.warning(
                                "Can save best model only with %s available, "
                                "skipping.",
                                metric,
                            )
                        else:
                            if monitor_op(current, best):
                                if self.verbose > 0:
                                    io_utils.print_msg(
                                        f"\nEpoch {epoch + 1}: {metric} "
                                        "improved "
                                        f"from {best:.5f} to {current:.5f}, "
                                        f"saving model"
                                    )
                                self.best[i] = current
                                improved_metrics.append(metric)
                            else:
                                if self.verbose > 0:
                                    io_utils.print_msg(
                                        f"\nEpoch {epoch + 1}: "
                                        f"{metric} did not improve "
                                        f"from {best:.5f}"
                                    )
                    if len(improved_metrics) > 0:
                        extra_name = '-'.join([self._removeprefix(s, 'val_') for s in improved_metrics])
                        path, extension = os.path.splitext(filepath)
                        filepath = f'{path}_improved{{{extra_name}}}{extension}'
                        model_to_save = getattr(self.model, self.sub_model_name) if self.sub_model_name is not None else self.model
                        if self.save_weights_only:
                            model_to_save.save_weights(
                                filepath,
                                overwrite=True,
                                options=self._options,
                            )
                        else:
                            model_to_save.save(
                                filepath,
                                overwrite=True,
                                include_optimizer=self.include_optimizer,
                                save_traces=False, # reduces a lot the saving time, but it requires to use custom_objects when loading the saved_model
                                options=self._options,
                            )

                else:
                    if self.verbose > 0:
                        io_utils.print_msg(
                            f"\nEpoch {epoch + 1}: saving model to {filepath}"
                        )
                    model_to_save = getattr(self.model, self.sub_model_name) if self.sub_model_name is not None else self.model
                    if self.save_weights_only:
                        model_to_save.save_weights(
                            filepath, overwrite=True, options=self._options
                        )
                    else:
                        model_to_save.save(
                            filepath, overwrite=True, include_optimizer=self.include_optimizer, save_traces=False, options=self._options
                        )

                self._maybe_remove_file()
            except IsADirectoryError:  # h5py 3.x
                raise IOError(
                    "Please specify a non-directory filepath for "
                    "ModelCheckpoint. Filepath used is an existing "
                    f"directory: {filepath}"
                )
            except IOError as e:  # h5py 2.x
                # `e.errno` appears to be `None` so checking the content of
                # `e.args[0]`.
                if "is a directory" in str(e.args[0]).lower():
                    raise IOError(
                        "Please specify a non-directory filepath for "
                        "ModelCheckpoint. Filepath used is an existing "
                        f"directory: f{filepath}"
                    )
                # Re-throw the error for any other causes.
                raise e

    def _removeprefix(self, s, pre):
        if pre and s.startswith(pre):
            return s[len(pre):]
        return s


class SampleImages(tf.keras.callbacks.Callback):
    def __init__(self, config, save_path='.', title='', image_num=2, plot_with_title=False, sample_ema=True, save_plot=True, show_plots=False, apply_standing_stats=False):
        self.config = config
        self.save_path = save_path
        self.title = title
        self.image_num = image_num
        self.plot_with_title = plot_with_title
        self.sample_ema = sample_ema
        self.save_plot = save_plot
        self.show_plots = show_plots
        self.current_epoch = 0
        self.apply_standing_stats = apply_standing_stats
        self.num_classes = config.DATA.num_classes
        
        self.num_col = self.num_classes if self.num_classes <= 10 else 10 # plot at most 10 classes
        
        tf.random.set_seed(self.config.RUN.seed)
        self.fixed_noise = tf.random.normal(shape=(self.image_num*self.num_col, self.config.MODEL.latent_dim))            
        self.fixed_labels = tf.tile(np.round(np.linspace(0, self.num_classes-1, self.num_col)).astype(int), [self.image_num]) # select the classes to plot
        self.fixed_labels = tf.cast(self.fixed_labels, tf.int32)

    def on_epoch_end(self, epoch, logs={}):
        epoch = self.current_epoch # optionally, remove this if the callback is used in a .fit() with more than 1 epoch
        
        generator = self.model.generator
        
        if self.sample_ema and hasattr(self.model, 'ema_generator'):
            if self.model.itrs > self.model.ema_start:
                #print("Sampling with EMA Generator")
                generator = self.model.ema_generator
                
        self.sample_images(model=generator, epoch=epoch+1)
        
        self.current_epoch += 1
        
    def sample_images(self, model, epoch):
            num_row = self.image_num
            num_col = self.num_col
            mpl.rcParams.update(mpl.rcParamsDefault)
            if(self.plot_with_title):
                fig, axes = plt.subplots(num_row, num_col, figsize=(15,1+1.5*self.image_num))
                title = f"{f'{self.title} - ' if self.title != '' else ''}Epoch: {epoch}"
                fig.suptitle(title)
            else:
                fig, axes = plt.subplots(num_row, num_col, figsize=(15,1.5*self.image_num))
                           
            if self.apply_standing_stats:
                model = misc.apply_standing_statistics(generator=model, standing_max_batch=self.config.DATA.batch_size, standing_step=self.config.DATA.batch_size,
                                                       latent_dim=self.config.MODEL.latent_dim, num_classes=self.num_classes, safe_copy=True, verbose=False)
                
            noise_and_labels = [self.fixed_noise, self.fixed_labels]
            fakes = model.predict(noise_and_labels, verbose=0)
            
            if np.min(fakes) < 0: # if images are normalized in range [-1,1] (i.e. using tanh as output function in the decoder)
                fakes = (fakes + 1) / 2

            for r in range(num_row):
                for c in range(num_col):
                    i = c + num_col * r
                    fake = fakes[i:i+1]
                    ax = axes[i//num_col, i%num_col]
                    ax.imshow(np.squeeze(fake,axis=(0)), cmap='gray')
                    ax.axis("off")

            if self.save_plot:
                os.makedirs(self.save_path, exist_ok=True) # this prevent an exception (i.e --> training stops) if the save folder has been deleted
                fig.savefig(self.save_path+'/'+'{:0>5}'.format(epoch)+'.png')
            if self.show_plots:
                plt.tight_layout()
                plt.show()

            plt.close(fig)


# Callback to fix the same seed at the beginning of each epoch
class FixSeed(tf.keras.callbacks.Callback):
    def __init__(self, seed):
        self.seed = 0
    
    def on_epoch_begin(self, epoch, logs=None):
        tf.random.set_seed(self.seed)


# I could use the built-in tf.keras.callbacks.History (which is automatically added to the callbacks list when calling model.fit())
# and convert the resulting History object to a pandas dataframe with pd.DataFrame.from_dict(history.history),
# but it won't properly log metrics that are added only at a certain epoch (e.g. FID, IS, etc. if start_eval_epoch > 0)
class HistoryDataframe(tf.keras.callbacks.Callback):
    def __init__(self):
        self.history = pd.DataFrame()
    
    def on_epoch_end(self, epoch, logs=None):
        if(self.history.empty):
            self.history = pd.DataFrame(columns=list(logs.keys()))
        new_row = dict(zip(list(logs.keys()), np.squeeze(list(logs.values()))))
        self.history = pd.concat([self.history, pd.DataFrame([new_row])], ignore_index=True)


# Inspired by https://github.com/keras-team/keras/blob/68f9af408a1734704746f7e6fa9cfede0d6879d8/keras/callbacks.py#L987
# but to check all the losses/metrics and not only the one called "loss"
class TerminateOnNaN(tf.keras.callbacks.Callback):
    """
    Callback that checks for NaN or Inf values in the losses and metrics during training.
    If NaN or Inf values are found, training is terminated.
    """
    def __init__(self, contains=''):
        """
        Initializes the NaNCheckCallback.

        Args:
        - contains: str, only check losses/metrics that contain this string in their name.
        """
        self.contains = contains

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            if self.contains in k:
                metric = tf_utils.sync_to_numpy_or_python_type(v)
                if np.isnan(metric) or np.isinf(metric):
                    io_utils.print_msg(
                        f"Batch {batch}: Invalid loss '{k}' with value '{metric}', terminating training"
                    )
                    self.model.stop_training = True
                    break

def lr_scheduler(milestones=[], gamma=0.1):
    """
    Returns a learning rate scheduler function that multiplies the learning rate by gamma
    at the specified milestones.

    Args:
        milestones (list): List of epoch or step indices at which to reduce the learning rate.
        gamma (float): Multiplicative factor by which to reduce the learning rate.

    Returns:
        A function that takes in the current epoch/step and learning rate, and returns the
        updated learning rate.
    """
    def _scheduler(step_or_epoch, lr):
        if step_or_epoch in milestones:
            return lr * gamma
        else:
            return lr
    return _scheduler