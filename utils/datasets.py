import tensorflow as tf
import numpy as np
from utils.cinic10 import cinic10
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image

tfk = tf.keras
tfkl = tf.keras.layers

class Dataset:
    def __init__(self, dataset_name, batch_size, val_batch_size=128, categorical=False,
                 val_categorical=False, normalize=True, dequantize=False, horizontal_flip=False,
                 resize=None, resize_size=None, padding=None, drop_remainder=False, cfgs=None, seed=None):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.categorical = categorical
        self.val_categorical = val_categorical
        self.normalize = normalize
        self.dequantize = dequantize
        self.horizontal_flip = horizontal_flip # only for training dataset
        self.resize = resize
        self.resize_size = resize_size
        self.padding = padding
        self.drop_remainder = drop_remainder
        self.cfgs = cfgs
        self.seed = seed

        self._load_config()
        self._set_cfgs_attributes()

    def _load_config(self):
        if self.dataset_name == 'FashionMNIST':
            self.num_channels = 1
            self.num_classes = 10
            self.image_size = 28
            self.resize = self.resize if self.resize is not None else False
            self.padding = self.padding if self.padding is not None else True
            self._load_data = tfk.datasets.fashion_mnist.load_data
        elif self.dataset_name == 'CIFAR10':
            self.num_channels = 3
            self.num_classes = 10
            self.image_size = 32
            self.resize = self.resize if self.resize is not None else False
            self.padding = self.padding if self.padding is not None else False
            self._load_data = tfk.datasets.cifar10.load_data  
        elif self.dataset_name == 'CIFAR100':
            self.num_channels = 3
            self.num_classes = 100
            self.image_size = 32
            self.resize = self.resize if self.resize is not None else False
            self.padding = self.padding if self.padding is not None else False
            self._load_data = tfk.datasets.cifar100.load_data
        elif self.dataset_name == 'CINIC10':
            self.num_channels = 3
            self.num_classes = 10
            self.image_size = 32
            self.resize = self.resize if self.resize is not None else False
            self.padding = self.padding if self.padding is not None else False
        elif self.dataset_name == 'DermaMNIST':
            self.num_channels = 3
            self.num_classes = 7
            self.image_size = 28
            self.resize = self.resize if self.resize is not None else True
            self.resize_size = self.resize_size if self.resize_size is not None else 32
            self.padding = self.padding if self.padding is not None else False
        else:
            raise ValueError('Invalid dataset name')
        
        assert not (self.resize and self.resize_size is None), "If resize is True, resize_size can't be None"
        assert not (self.padding and self.resize_size <= self.image_size), "If padding is True, resize_size must be greater than image_size"
        assert (self.resize and self.padding) == False, "Resize and padding can't be done togheter"

        self.resize = self.resize if self.resize_size != self.image_size else False
        if self.resize:
            self.image_size = self.resize_size
        
        if self.padding:
            self.padding_size = int((self.resize_size - self.image_size) / 2)
            self.image_size = self.image_size + self.padding_size  * 2
                
    def load_dataset(self, splits=['train', 'val', 'test'], merge_train_val=False, verbose=True):
        self.X_train = self.y_train = self.X_val = self.y_val = self.X_test = self.y_test = None
        training_dataset = validation_dataset = test_dataset = None

        ##### SUPPORTED DATASETS #####
        print("Loading dataset: ", self.dataset_name)
        if self.dataset_name == 'CINIC10':
            cinic10_ds = cinic10()
            X_train, y_train, X_val, y_val, X_test, y_test = cinic10_ds.loadData("./datasets/", oneHot=False, num_workers=8)
            X_train_val = np.vstack([X_train, X_val])
            y_train_val = np.hstack([y_train, y_val])
            X_train_val, y_train_val = shuffle(X_train_val, y_train_val, random_state=0)
            X_test, y_test = shuffle(X_test, y_test, random_state=0) # could be useful since the dataset is ordered by class
            if merge_train_val:
                X_train = X_train_val
                y_train = y_train_val
            else:
                # Generate a larger training dataset and a smaller val dataset than the original ones
                X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=10000, random_state=0, stratify=y_train_val)
        elif self.dataset_name == 'DermaMNIST':
            tf.keras.utils.get_file(
                fname="dermamnist.npz",
                origin="https://zenodo.org/record/6496656/files/dermamnist.npz?download=1",
                cache_dir=".",
                cache_subdir="datasets/MedMNIST",
                extract=False
            )
            med_ds = np.load('./datasets/MedMNIST/dermamnist.npz')
            X_train, y_train, X_val, y_val, X_test, y_test = med_ds['train_images'], med_ds['train_labels'], med_ds['val_images'], \
                                                            med_ds['val_labels'], med_ds['test_images'], med_ds['test_labels']
            if merge_train_val:
                X_train = np.vstack([X_train, X_val])
                y_train = np.vstack([y_train, y_val])
        else:
            (X_train_val, y_train_val), (X_test, y_test) = self._load_data()
            if merge_train_val:
                X_train = X_train_val
                y_train = y_train_val
            else:
                X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=len(X_test), random_state=0, stratify=y_train_val)

        ##### RESIZE #####
        if self.resize and not self.padding:
            print("    Resizing images to size: ", self.image_size)
            X_train_temp, X_val_temp, X_test_temp = [], [], []

            if 'train' in splits:
                for image_np in X_train:
                    im = Image.fromarray(image_np)
                    im = im.resize((self.resize_size,self.resize_size), resample=Image.Resampling.LANCZOS)
                    im = np.array(im)
                    X_train_temp.append(im)
                X_train = np.array(X_train_temp)

            if 'val' in splits and not merge_train_val:
                for image_np in X_val:
                    im = Image.fromarray(image_np)
                    im = im.resize((self.resize_size,self.resize_size), resample=Image.Resampling.LANCZOS)
                    im = np.array(im)
                    X_val_temp.append(im)
                X_val = np.array(X_val_temp)

            if 'test' in splits:
                for image_np in X_test:
                    im = Image.fromarray(image_np)
                    im = im.resize((self.resize_size,self.resize_size), resample=Image.Resampling.LANCZOS)
                    im = np.array(im)
                    X_test_temp.append(im)
                X_test = np.array(X_test_temp)

        if self.padding:
            print("    Padding images to size: ", self.image_size)

        ##### TRAINING SET #####
        print("    Building Tensorflow dataset")
        if 'train' in splits:
            X_train = X_train.astype("float32") / 255.0
            X_train = np.reshape(X_train, (-1, X_train.shape[1], X_train.shape[2], self.num_channels))
            if self.padding:
                # Avoid using tfkl.ZeroPadding2D to not occupy GPU memory at this stage
                X_train = np.pad(X_train, ((0,0), (self.padding_size,self.padding_size), (self.padding_size,self.padding_size), (0,0))) # tfkl.ZeroPadding2D(padding=(self.padding_size, self.padding_size))(X_train)
            if self.normalize:
                # Avoid using tfkl.Normalization to not occupy GPU memory at this stage
                X_train = (X_train - 0.5) / 0.5 # tfkl.Normalization(mean=[0.5], variance=[0.5**2])(X_train).numpy()
                if self.dequantize:
                    X_train = X_train + tf.random.uniform(shape=X_train.shape, minval=0.0, maxval=1./127.5)
            if self.categorical:
                y_train = tfk.utils.to_categorical(y_train, self.num_classes)
            else:
                if tf.rank(y_train) == 1:
                    y_train = tf.cast(tf.expand_dims(y_train, -1), tf.int32)
            training_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            training_dataset = training_dataset.cache().shuffle(1024)
            if self.horizontal_flip:
                f = lambda image, label: (tf.image.random_flip_left_right(image), label)
                training_dataset = training_dataset.map(f, num_parallel_calls=tf.data.AUTOTUNE)
            training_dataset = training_dataset.batch(self.batch_size, drop_remainder=self.drop_remainder).prefetch(buffer_size=tf.data.AUTOTUNE)
            self.X_train = X_train
            self.y_train = y_train

        ##### VALIDATION SET #####
        if 'val' in splits and not merge_train_val:
            X_val = X_val.astype("float32") / 255.0
            X_val = np.reshape(X_val, (-1, X_val.shape[1], X_val.shape[2], self.num_channels))
            if self.padding:
                X_val = np.pad(X_val, ((0,0), (self.padding_size,self.padding_size), (self.padding_size,self.padding_size), (0,0))) # tfkl.ZeroPadding2D(padding=(self.padding_size, self.padding_size))(X_val)
            if self.normalize:
                X_val = (X_val - 0.5) / 0.5 # tfkl.Normalization(mean=[0.5], variance=[0.5**2])(X_val).numpy()
            if self.val_categorical:
                y_val = tfk.utils.to_categorical(y_val, self.num_classes)
            else:
                if tf.rank(y_val) == 1:
                    y_val = tf.cast(tf.expand_dims(y_val, -1), tf.float32)
            validation_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
            validation_dataset = validation_dataset.batch(self.val_batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
            self.X_val = X_val
            self.y_val = y_val

        if merge_train_val:
            self.X_val = None
            self.y_val = None

        ##### TEST SET #####
        if 'test' in splits:
            X_test = X_test.astype("float32") / 255.0
            X_test = np.reshape(X_test, (-1, X_test.shape[1], X_test.shape[2], self.num_channels))
            if self.padding:
                X_test = np.pad(X_test, ((0,0), (self.padding_size,self.padding_size), (self.padding_size,self.padding_size), (0,0))) # tfkl.ZeroPadding2D(padding=(self.padding_size, self.padding_size))(X_test)
            if self.normalize:
                X_test = (X_test - 0.5) / 0.5 # tfkl.Normalization(mean=[0.5], variance=[0.5**2])(X_test).numpy()
            if self.val_categorical:
                y_test = tfk.utils.to_categorical(y_test, self.num_classes)
            else:
                if tf.rank(y_test) == 1:
                    y_test = tf.cast(tf.expand_dims(y_test, -1), tf.float32)
            test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
            test_dataset = test_dataset.batch(self.val_batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
            self.X_test = X_test
            self.y_test = y_test

        if verbose:
            if 'train' in splits:
                print("Training images shape", X_train.shape)
                print("Training labels shape", y_train.shape)
            if 'val' in splits and not merge_train_val:
                print("Validation images shape", X_val.shape)
                print("Validation labels shape", y_val.shape)
            if 'test' in splits:
                print("Test images shape", X_test.shape)
                print("Test labels shape", y_test.shape)

        # set self.image_size looking at the actual shape just to be sure
        if 'train' in splits:
            self.image_size = X_train.shape[1]
        elif 'val' in splits and not merge_train_val:
            self.image_size = X_val.shape[1]
        elif 'test' in splits:
            self.image_size = X_test.shape[1]
        
        self.X_train_len = len(X_train) # set this property to have access to this information even when 'train' is not in the desired splits
        self._set_cfgs_attributes()

        return training_dataset, validation_dataset, test_dataset

    def get_numpy_splits(self):
        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
        
    def _set_cfgs_attributes(self):
        self.cfgs.DATA.image_size = self.image_size
        self.cfgs.DATA.num_classes = self.num_classes
        self.cfgs.DATA.num_channels = self.num_channels