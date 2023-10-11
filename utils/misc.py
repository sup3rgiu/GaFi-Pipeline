import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_addons as tfa # needed to load models that used optimizers/layers from this package. Do not remove even if happears as unused!
import numpy as np
import pandas as pd
import json
import shutil
import random
import os
import ast
import types
import glob
from tqdm.auto import tqdm
from datetime import datetime
import subprocess as sp

from models.layers import SDense, ConditionalBatchNorm, Conv, SelfAttention
from utils.callbacks import HistoryDataframe

tfkl = tf.keras.layers

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)

def check_and_fix_duplicate_run(run_name, base_path):
    orig_save_path = os.path.join(base_path, run_name)
    if os.path.exists(orig_save_path):
        now = datetime.now()
        now_string = now.strftime("%Y%m%d%H%M%S")
        run_name = run_name + '_' + now_string
        print(f"Save path '{orig_save_path}' already exists. Adding temp name '{now_string}' to the run name")
    return run_name

def load_classifier(classifier_type='resnet20', dataset='CIFAR10', optimizer='sgdw', random_erasing=True, padding=False, image_size=32, resize=False, normalize=False, extra_name='', cls_save_path='./save/Models/Classifiers'):

    classifier_name = classifier_type.lower()

    name_pretrained = f"""pretrained_{classifier_name}_{dataset.lower()}_{optimizer}{f'_pad{image_size}' if padding else ''}\
{f'_resized{image_size}' if resize else ''}{'_normalized' if normalize else ''}{'_erasing' if random_erasing else ''}{f'_{extra_name}' if extra_name != '' else ''}"""

    print(f"Loading pretrained classifier \"{name_pretrained}\"")

    path = os.path.join(cls_save_path, dataset, name_pretrained)
    c_criterion = lambda y_true, y_pred: tf.reduce_mean(tfk.losses.categorical_crossentropy(y_true, y_pred)) 
    classifier_pretrained = tfk.models.load_model(path, custom_objects={'<lambda>': c_criterion}) # custom_object needed if the classifier was trained with the loss specified in this way
    classifier_pretrained.trainable = False

    return classifier_pretrained

def reload_gan_model(gan, model_save_path, backup_files=True, callbacks=None):
    custom_objects={"SDense": SDense, "ConditionalBatchNorm": ConditionalBatchNorm, "Conv": Conv, "SelfAttention": SelfAttention}
    generator = tfk.models.load_model(os.path.join(model_save_path, 'generator'), custom_objects=custom_objects)
    discriminator = tfk.models.load_model(os.path.join(model_save_path, 'discriminator'), custom_objects=custom_objects)
    ema_generator = tfk.models.load_model(os.path.join(model_save_path, 'ema_generator'), custom_objects=custom_objects)
    gan.generator = generator
    gan.discriminator = discriminator
    gan.ema_generator = ema_generator
    gan.g_optimizer = generator.optimizer or gan.g_optimizer
    gan.d_optimizer = discriminator.optimizer or gan.d_optimizer
  
    with open(os.path.join(model_save_path, 'config.json')) as config_file:
        config = json.load(config_file, object_hook=lambda d: types.SimpleNamespace(**d))
    
    history = pd.read_csv(os.path.join(model_save_path, 'history.csv'))
    
    restored_epoch = len(history)
    
    basket_size = config.OPTIMIZATION.batch_size * config.OPTIMIZATION.d_updates_per_step * config.OPTIMIZATION.acml_steps
    restore_itr = restored_epoch * (config.DATA.train_size // basket_size)

    gan.itrs.assign(restore_itr)
    gan.current_eval_epoch = restored_epoch
    
    if callbacks is not None:
        for cb in callbacks:
            if hasattr(cb, 'current_epoch'):
                cb.current_epoch = restored_epoch
            if isinstance(cb, HistoryDataframe):
                cb.history = history
                
    if backup_files:
        source = model_save_path
        destination = os.path.join(model_save_path, f'backup_epoch_{restored_epoch}')
        allfiles = os.listdir(source)
        files_to_backup = [x for x in allfiles if 'backup_' not in x] # do not move in the backup folder the a already existing backup folder
        for f in files_to_backup:
            src_path = os.path.join(source, f)
            dst_path = os.path.join(destination, f)
            shutil.move(src_path, dst_path)
    
    return {'gan': gan,
            'history': history,
            'epoch': restored_epoch}

def load_model(model_path, verbose=False):
    orig_model_path = model_path
    model_path = glob.glob(model_path)
    if len(model_path) > 0:
        if len(model_path) > 1:
            print(f"Multiple models found for the given filename pattern matching, loading the first one")
        model_path = model_path[0]
    else:
        if verbose:
            print(f"Model not found at path {orig_model_path}")
        return None

    if verbose:
        print(f"Loading model from '{model_path}'")
    custom_objects={"SDense": SDense, "ConditionalBatchNorm": ConditionalBatchNorm, "Conv": Conv, "SelfAttention": SelfAttention}
    model = tfk.models.load_model(model_path, custom_objects=custom_objects)

    return model

def reset_bn_stats(model):
    for layer in model.layers:
        if isinstance(layer, ConditionalBatchNorm):
            layer.reset_standing_stats()
        elif isinstance(layer, tfkl.BatchNormalization):
            layer.moving_mean.assign(tf.zeros_like(layer.moving_mean))
            layer.moving_variance.assign(tf.ones_like(layer.moving_variance)) 
            
def apply_standing_statistics(generator, standing_max_batch, standing_step, latent_dim, num_classes, safe_copy=False, verbose=False):
    if verbose:
        print(f"Applying standing stats {'(safely, on a copy of the given generator)' if safe_copy else ''}...")

    # Make a copy to preserve the original model
    if safe_copy:
        gen_copy = tfk.models.clone_model(generator)
        gen_copy.set_weights(generator.get_weights())
        generator = gen_copy

    reset_bn_stats(generator)
    for i in tqdm(range(standing_step), disable=not verbose):
        rand_batch_size = random.randint(1, standing_max_batch)
        noise = tf.random.normal(shape=(rand_batch_size, latent_dim))
        labels = tf.random.uniform(shape=(rand_batch_size,), minval=0, maxval=num_classes, dtype=tf.int32)
        fake_images = generator([noise, labels], training=True) # training=True important! Needed to accumulate moving_mean and moving_var in BN layers

    if safe_copy:
        return generator

def generate_dataset(model, num_classes, filter=False, class_samples=5000, batch_size=128, classifier_pretrained=None, threshold=0.0, stddev=1.0, filtering_attempts=-1, seed=None, verbose=False):
    if seed is None:
        tf.random.set_seed(random.randint(1,10000))
    else:
        tf.random.set_seed(seed)
    fake_dataset = []
    fake_labels = []
    
    filtered_count = 0

    for c in tqdm(range(num_classes), disable=not verbose, bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}', desc='Fake dataset generation'):
        if filter:
            _fake = []
            i = 0
            attempts = 0
            while i < class_samples:
                # Calculate the number of samples that still need to be generated for the class
                # taking into account the number of samples already generated.
                samples_to_generate = (class_samples - i) * (1 + (i != 0)) # TODO: improve this heuristic
                _stddev = stddev
                if isinstance(stddev, (tuple, list)):
                    _stddev = tf.random.uniform(shape=(samples_to_generate, 1), minval=stddev[0], maxval=stddev[1])
                interpolation_noise = tf.random.normal(shape=(samples_to_generate, model.input[0].shape[1]), stddev=_stddev)
                sampling_label = tf.cast([c], tf.int32)
                sampling_label = tf.repeat(sampling_label,samples_to_generate,axis=0)
                noise_and_labels = [interpolation_noise, sampling_label]
                fake = model.predict(noise_and_labels, verbose=0, batch_size=batch_size)

                # Filter the generated samples, removing those that do not respect the filtering rule
                pred = classifier_pretrained.predict(fake, verbose=0, batch_size=batch_size)
                mask = tf.logical_and(tf.argmax(pred, -1) == c, tf.greater(pred, threshold)[:, c])
                fake = fake[mask]
                filtered_count += pred.shape[0] - fake.shape[0] # For stats purposes
                _fake.append(fake)
                i += fake.shape[0]
                attempts += 1

                if filtering_attempts > 0 and attempts >= filtering_attempts:
                    fake = model.predict(noise_and_labels, verbose=0, batch_size=batch_size)
                    _fake.append(fake)
                    i += fake.shape[0]
                    if verbose:
                        print(f"Adding unfiltered images for class: {c}")
            fake = np.vstack(_fake)
            fake = fake[:class_samples]
        else:
            if isinstance(stddev, (tuple, list)):
                stddev = tf.random.uniform(shape=(class_samples, 1), minval=stddev[0], maxval=stddev[1])
            interpolation_noise = tf.random.normal(shape=(class_samples, model.input[0].shape[1]), stddev=stddev)
            sampling_label = tf.cast([c], tf.int32)
            sampling_label = tf.repeat(sampling_label, class_samples, axis=0)
            noise_and_labels = [interpolation_noise, sampling_label]
            fake = model.predict(noise_and_labels, verbose=0, batch_size=batch_size)
        
        fake_dataset.append(fake)
        fake_labels.append(np.ones(class_samples)*c)
        
    #return fake_dataset, fake_labels
    fake_dataset = np.vstack(fake_dataset)
    fake_labels = np.array(fake_labels)
    fake_labels = np.reshape(fake_labels, (fake_labels.shape[0]*fake_labels.shape[1]))
    fake_labels = tfk.utils.to_categorical(fake_labels, num_classes)
    
    if filter and verbose:
        print(f"Total filtered images: {filtered_count}")

    return fake_dataset, fake_labels

# Use this function if we want to subsample a dataset keeping it perfectly balanced
def subsample_balanced_dataset(dataset, labels, size, return_labels=False):
    if tf.rank(labels) == 2:
        labels = tf.argmax(labels, -1)
    num_classes = tf.shape(tf.unique(labels)[0])[0]
    samples_per_class = size // num_classes
    assert samples_per_class * num_classes == size, "Size must be divisible by num_classes"
    images = []
    new_labels = []
    for i in range(num_classes):
        images_class = tf.gather(dataset, tf.reshape(tf.where(labels == i), -1)) # labels in one-hot
        images_class = images_class[:samples_per_class]
        images.append(images_class)
        new_labels.append(tf.ones((samples_per_class,)) * i)
    images = tf.concat(images, 0)
    new_labels = tf.concat(new_labels, 0)
    if return_labels:
        return images, new_labels
    else:
        return images
    
def subsample_dataset(X, y, samples_per_class):
    if samples_per_class == 0:
        return None, None
    indices = []
    y_argmax = np.argmax(y, axis=1)
    for c in np.unique(y_argmax):
        indx = np.random.permutation(np.where(y_argmax == c)[0])[:samples_per_class]
        indices.append(indx)
    indices = np.hstack(indices)
    return X[indices], y[indices]

def linspace(start, stop, step=1):
    """
    Returns evenly spaced numbers over a specified interval.

    Args:
        start (int, float): The starting value of the sequence.
        stop (int, float): The end value of the sequence.
        step (int, float, optional): The spacing between the values. Defaults to 1.

    Returns:
        list: A list of evenly spaced values between start and stop.
    """
    res = []
    val = type(step)(start) # cast start to the type of step
    while val <= stop:
        res.append(val)
        val += step
        val = round(val, 3)
    return res

def get_list_or_tuple(v, type_check=None):
    """
    Converts the given object to a list of integers/floats, if possible.
    If the given argument is a list of integers/floats, it is returned as is.
    If the given argument is a tuple (triplet) of integers/floats, it is converted to a list of integers/floats thanks to the linspace function.
    If the given argument is a string representing a triplet of integers/floats, it is converted to a tuple and then to a list of integers/floats thanks to the linspace function (needed when a triplet is given in the yaml file).

    Args:
        v (list, tuple, str): The object to be converted.
        type_check (type, optional): if specified, checks the type of the elements of the list/typle. Defaults to None.

    Returns:
        list: The object as a list of integers/floats.

    Raises:
        ValueError: If the given object is not in a supported format.
    """
    if isinstance(v, list):
        if type_check is not None:
            if not all(isinstance(x, type_check) for x in v):
                raise ValueError(f"The given list contains non-{type_check.__name__} values")
        ckpt_epochs = v
    elif isinstance(v, tuple):
        pass
    elif isinstance(v, str):
        try:
            v = ast.literal_eval(v)
            if not isinstance(v, tuple):
                raise ValueError
        except ValueError:
            raise ValueError("The given object is not in a supported format")
    else:
        raise ValueError("The given object is not in a supported format")

    if isinstance(v, tuple):
        if type_check is not None:
            if not all(isinstance(x, type_check) for x in v):
                raise ValueError(f"The given tuple contains non-{type_check.__name__} values")
        if len(v) != 3:
            raise ValueError("The supplied tuple must have 3 values: (start, end, step)")
        ckpt_epochs = linspace(v[0], v[1], v[2])

    return ckpt_epochs

def pretty_print(s, separator='#', spacing=''):
    """
    Prints a string with a separator above and below it, centered and padded with spaces.

    Args:
        s (str): The string to be printed.
        separator (str, optional): The character to be used as separator. Defaults to '#'.
        spacing (str, optional): The spacing to be added before and after the separator. Defaults to ''.
    """
    min_spacing = 3
    center_len = 3
    line_len = max(len(s) + (center_len+min_spacing)*2, 30)
    spacing_len = (line_len - len(s) - center_len*2) // 2
    
    line_len = len(s) + (spacing_len + center_len) * 2
    
    result = spacing
    result += separator * line_len
    result += '\n' + separator*center_len + ' '*spacing_len + s + ' '*spacing_len + separator*center_len +'\n'
    result += separator * line_len
    result += spacing
    print(result)

def print_gpu_usage():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    COMMAND = 'nvidia-smi --query-gpu=memory.used --format=csv'
    out = sp.check_output(COMMAND.split())
    out = ' '.join(output_to_list(out))
    print(out)