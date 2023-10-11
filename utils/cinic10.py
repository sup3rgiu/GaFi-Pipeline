import tensorflow as tf
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import imageio
from six.moves import urllib
import tarfile
import multiprocessing

# Modified from https://git.altinel.dev/fazil/CINIC-10-TFLoader and https://github.com/One-sixth/simple-cinic-10-dataset-loader

class cinic10:
    labelDict = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3,
                 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8,
                 'truck': 9}

    def loadData(self, pathToDatasetFolder, oneHot=False, num_workers=8):
        """
        pathToDatasetFolder: Parent folder of CINIC-10 dataset folder or CINIC-10.tar.gz file
        oneHot: Label encoding (one hot encoding or not)
        Return: Train, validation and test sets and label numpy arrays
        """
        sourceUrl = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"
        pathToFile = self.downloadDataset(pathToDatasetFolder, "CINIC-10.tar.gz", sourceUrl)

        pathToTrain = os.path.join(pathToFile, "train")
        pathToVal = os.path.join(pathToFile, "valid")
        pathToTest = os.path.join(pathToFile, "test")

        imgPathsTrain = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(pathToTrain)) for f in fn]
        imgPathsVal = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(pathToVal)) for f in fn]
        imgPathsTest = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(pathToTest)) for f in fn]

        npz_path = os.path.join(pathToDatasetFolder, 'CINIC-10', 'cinic10.npz')
        if os.path.exists(npz_path):
            print("Loading from npz")
            self.load_from_npz(npz_path)
        else:
            print("Loading")
            img_train_classes = list(map(lambda p: self.labelDict[os.path.basename(os.path.dirname(p))], imgPathsTrain))
            self.X_train, self.y_train = self._read_images(imgPathsTrain, img_train_classes, num_workers=num_workers)

            img_val_classes = list(map(lambda p: self.labelDict[os.path.basename(os.path.dirname(p))], imgPathsVal))
            self.X_val, self.y_val = self._read_images(imgPathsVal, img_val_classes, num_workers=num_workers)

            img_test_classes = list(map(lambda p: self.labelDict[os.path.basename(os.path.dirname(p))], imgPathsTest))
            self.X_test, self.y_test = self._read_images(imgPathsTest, img_test_classes, num_workers=num_workers)

            print("Saving to npz to improve loading speed")
            self.save_npz(npz_path)
            print("+ Saved")

        if oneHot:
            self.y_train = toOneHot(self.y_train, 10)
            self.y_val = toOneHot(self.y_val, 10)
            self.y_test = toOneHot(self.y_test, 10)

        print("+ Dataset loaded")

        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
    
    
    def data(self):
        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test


    def _read_images(self, img_paths, img_classes, num_workers=8):
        _cpu_count = num_workers # multiprocessing.cpu_count()

        with multiprocessing.Pool(processes=_cpu_count) as pool:
            results = pool.starmap(_read_image, zip(img_paths, img_classes))
        
        images, classes = [np.array(a) for a in zip(*results)] # tuples to Numpy arrays
        return images, classes
    
    
    def load_from_npz(self, npz_file='cinic10.npz'):
        '''
        load dataset from npz. more fast than load from dir
        :param npz_file:
        :return:
        '''
        d = np.load(npz_file)
        self.X_train = d['X_train']
        self.y_train = d['y_train']
        self.X_val = d['X_val']
        self.y_val = d['y_val']
        self.X_test = d['X_test']
        self.y_test = d['y_test']


    def save_npz(self, npz_file='cinic10.npz'):
        '''
        save all data in npz to improve loading speed
        :param npz_file:
        :return:
        '''
        np.savez_compressed(npz_file, X_train=self.X_train, y_train=self.y_train, X_val=self.X_val,
                            y_val=self.y_val, X_test=self.X_test, y_test=self.y_test)


    def downloadDataset(self, dirName, fileName, sourceUrl):
        """
        https://github.com/tflearn/tflearn/blob/master/tflearn/datasets/cifar10.py
        """
        cinicDirName = os.path.join(dirName, "CINIC-10/")
        if not os.path.exists(cinicDirName):
            pathToFile = os.path.join(dirName, fileName)
            if not os.path.exists(pathToFile):
                print("Downloading")
                pathToFile, _ = urllib.request.urlretrieve(sourceUrl, pathToFile, DLProgbar())
                #pathToFile, _ = urllib.request.urlretrieve(sourceUrl, pathToFile, reporthook)
                print("+ Downloaded")
            untar(pathToFile, cinicDirName, delete=True)
        else:
            print("+ Dataset already downloaded")
        return cinicDirName


def _read_image(path, img_class):
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.tile(img[..., None], [1, 1, 3])
    return img, img_class

class DLProgbar:
    """
    Manage progress bar state for use in urlretrieve.
    https://github.com/keras-team/keras/blob/68f9af408a1734704746f7e6fa9cfede0d6879d8/keras/utils/data_utils.py#L321
    """

    def __init__(self):
        self.progbar = None
        self.finished = False

    def __call__(self, block_num, block_size, total_size):
        if not self.progbar:
            if total_size == -1:
                total_size = None
            self.progbar = tf.keras.utils.Progbar(total_size)
        current = block_num * block_size

        if total_size is None:
            self.progbar.update(current)
        else:
            if current < total_size:
                self.progbar.update(current)
            elif not self.finished:
                self.progbar.update(self.progbar.target)
                self.finished = True


def reporthook(blocknum, blocksize, totalsize):
    """
    reporthook from stackoverflow #13881092
    https://github.com/tflearn/tflearn/blob/master/tflearn/datasets/cifar10.py
    """
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize:  # near the end
            sys.stderr.write("\n")
    else:  # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))


def untar(fname, path, delete=False):
    if (fname.endswith("tar.gz")):
        print("Extracting tar file")
        os.mkdir(path)
        tar = tarfile.open(fname)
        tar.extractall(path=path)
        tar.close()
        print("+ Extracted")
        if delete:
            os.remove(fname)
            print("+ Tar file deleted")
    else:
        print("Not a tar.gz file")


def toOneHot(y, nb_classes=None):
    """
    https://github.com/tflearn/tflearn/blob/master/tflearn/data_utils.py#L36
    """
    if nb_classes:
        # y = np.asarray(y, dtype='int32')
        if len(y.shape) > 2:
            print("Warning: data array ndim > 2")
        if len(y.shape) > 1:
            y = y.reshape(-1)
        Y = np.zeros((len(y), nb_classes))
        Y[np.arange(len(y)), y] = 1.
        return Y
    else:
        y = np.array(y)
        return (y[:, None] == np.unique(y)).astype(np.float32)