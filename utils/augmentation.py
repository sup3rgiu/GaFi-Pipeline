import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
import keras
import keras_cv
import numpy as np
from keras.layers.preprocessing.image_preprocessing import get_zoom_matrix
from keras.layers.preprocessing import preprocessing_utils

tfk = tf.keras
tfkl = tf.keras.layers

# https://github.com/tensorflow/models/blob/dc3c6706f09fef813978ec5efe608aa3af00022c/official/vision/ops/augment.py#L314
# BUT FIXED TO SUPPORT GRAYSCALE IMAGES --> mask = tf.tile(mask, [1, 1, image_channels]) in _fill_rectangle
# AND TO RESCALE THE RANDOM NOISE IN THE ORIGINAL IMAGE RANGE
def _fill_rectangle(image,
                    center_width,
                    center_height,
                    half_width,
                    half_height,
                    replace=None):
    """Fills blank area."""
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    image_channels = tf.shape(image)[2]

    lower_pad = tf.maximum(0, center_height - half_height)
    upper_pad = tf.maximum(0, image_height - center_height - half_height)
    left_pad = tf.maximum(0, center_width - half_width)
    right_pad = tf.maximum(0, image_width - center_width - half_width)

    cutout_shape = [
      image_height - (lower_pad + upper_pad),
      image_width - (left_pad + right_pad)
    ]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = tf.pad(
      tf.zeros(cutout_shape, dtype=image.dtype),
      padding_dims,
      constant_values=1)
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, image_channels])
    
    if replace is None:
        fill = tf.random.normal(tf.shape(image), dtype=image.dtype)
        image_max = tf.reduce_max(image)
        image_min = tf.reduce_min(image)
        fill_max = tf.reduce_max(fill)
        fill_min = tf.reduce_min(fill)
        fill = (image_max - image_min) * (fill - fill_min) / (fill_max - fill_min) + image_min
    elif isinstance(replace, tf.Tensor):
        fill = replace
    else:
        fill = tf.ones_like(image, dtype=image.dtype) * replace
    image = tf.where(tf.equal(mask, 0), fill, image)

    return image


# https://github.com/keras-team/keras-cv/blob/v0.3.4/keras_cv/layers/preprocessing/random_cutout.py#L155
# BUT FIXED TO RESCALE THE RANDOM NOISE IN THE ORIGINAL IMAGE RANGE
def _compute_rectangle_fill(self, inputs):
    input_shape = tf.shape(inputs)
    if self.fill_mode == "constant":
        fill_value = tf.fill(input_shape, self.fill_value)
    else:
        # gaussian noise
        fill_value = tf.random.normal(input_shape)
        image_max = tf.reduce_max(inputs)
        image_min = tf.reduce_min(inputs)
        fill_max = tf.reduce_max(fill_value)
        fill_min = tf.reduce_min(fill_value)
        fill_value = (image_max - image_min) * (fill_value - fill_min) / (fill_max - fill_min) + image_min
    return fill_value


# https://github.com/keras-team/keras-cv/blob/v0.3.4/keras_cv/layers/preprocessing/base_image_augmentation_layer.py#L389
# BUT FIXED TO SUPPORT THE autocast OPTION WHICH IS AN ACCEPTED VALUE FOR THE **kwargs OF THE BASE tf.keras.layers.Layer CLASS
# IF 'autocast' IS SET TO True DURING THE LAYER CONSTRUCTION, THEN THE INPUTS ARE CASTED TO THE COMPUTE DTYPE OF THE LAYER
def _ensure_inputs_are_compute_dtype(self, inputs):
    IMAGES = "images"
    if self._autocast:
        if isinstance(inputs, dict):
            inputs[IMAGES] = keras_cv.utils.preprocessing.ensure_tensor(
                inputs[IMAGES],
                self.compute_dtype,
            )
        else:
            inputs = keras_cv.utils.preprocessing.ensure_tensor(
                inputs,
                self.compute_dtype,
            )
    return inputs


# Speed up https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomRotation applying the transformation directly on the whole batch
class RandomRotationF(tfkl.Layer):
    def __init__(self, factor_degree, interpolation='bilinear', fill_mode='constant', fill_value=0.0, name=None, **kwargs):
              
        self.factor_degree = factor_degree
        
        if isinstance(factor_degree, (tuple, list)) and len(factor_degree) == 2:
            self.lower = factor_degree[0] * np.pi/180
            self.upper = factor_degree[1] * np.pi/180
        else:
            self.lower = -factor_degree * np.pi/180
            self.upper = factor_degree * np.pi/180
            
        self.interpolation = interpolation
        self.fill_mode = fill_mode
        self.fill_value = fill_value

        super(RandomRotationF, self).__init__(name=name, **kwargs)

    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()

        def random_rotate():
            rotation_angles = tf.random.uniform((tf.shape(inputs)[0],), minval=self.lower, maxval=self.upper, dtype=tf.float32)
            return tfa.image.rotate(inputs, angles=rotation_angles, interpolation=self.interpolation, fill_mode=self.fill_mode, fill_value=self.fill_value)

        output = tf.cond(tf.equal(training, True), random_rotate, lambda: inputs)
        output.set_shape(inputs.shape)
        return output
   
    def get_config(self):
        config = super().get_config()
        config.update({
            "factor_degree": self.factor_degree,
            "interpolation": self.interpolation,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
        })
        return config
    

# Speed up https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomZoom applying the transformation directly on the whole batch
# We could speed up natively the tfkl.RandomZoom layer setting 'self.auto_vectorize = False', but it's still slower than this implementation
def zoom(
    images,
    height_factor,
    width_factor = None,
    interpolation: str = "nearest",
    fill_mode: str = "constant",
    name = None,
    fill_value = 0.0,
):
    
    if isinstance(height_factor, (tuple, list)):
        height_lower = height_factor[0]
        height_upper = height_factor[1]
    else:
        height_lower = -height_factor
        height_upper = height_factor
        
    if width_factor is not None:
        if isinstance(width_factor, (tuple, list)):
            width_lower = width_factor[0]
            width_upper = width_factor[1]
        else:
            width_lower = -width_factor
            width_upper = width_factor
    
    images = preprocessing_utils.ensure_tensor(images, dtype=None)
    original_shape = images.shape
    images_shape = tf.shape(images)
    batch_size = images_shape[0]
    img_hd = tf.cast(images_shape[-3], tf.float32)
    img_wd = tf.cast(images_shape[-2], tf.float32)

    height_zoom = tf.random.uniform(
        shape=[batch_size, 1],
        minval=1.0 + height_lower,
        maxval=1.0 + height_upper,
    )
    if width_factor is not None:
        width_zoom = tf.random.uniform(
            shape=[batch_size, 1],
            minval=1.0 + width_lower,
            maxval=1.0 + width_upper,
        )
    else:
        width_zoom = height_zoom

    zooms = tf.cast(tf.concat([width_zoom, height_zoom], axis=1), dtype=tf.float32)
    
    output = keras.layers.preprocessing.image_preprocessing.transform(
        images,
        get_zoom_matrix(zooms, img_hd, img_wd),
        fill_mode=fill_mode,
        fill_value=fill_value,
        interpolation=interpolation,
    )
    return output

class RandomZoomF(tfkl.Layer):
    def __init__(self, height_factor, width_factor=None, interpolation='bilinear', fill_mode='constant', fill_value=0.0, name=None, **kwargs):
                
        self.height_factor = height_factor
        self.width_factor = width_factor
        self.interpolation = interpolation
        self.fill_mode = fill_mode
        self.fill_value = fill_value

        super(RandomZoomF, self).__init__(name=name, **kwargs)

    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()

        def random_zoom():
            return zoom(inputs, height_factor=self.height_factor, width_factor=self.width_factor, 
                        interpolation=self.interpolation, fill_mode=self.fill_mode, fill_value=self.fill_value)
        
        output = tf.cond(tf.equal(training, True), random_zoom, lambda: inputs)
        output.set_shape(inputs.shape)
        return output
   
    def get_config(self):
        config = super().get_config()
        config.update({
            "height_factor": self.height_factor,
            "width_factor": self.width_factor,
            "interpolation": self.interpolation,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
        })
        return config


# Speed up https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomFlip applying the transformation directly on the whole batch
class RandomFlipF(tfkl.Layer):
    def __init__(self, mode='horizontal_and_vertical', name=None, **kwargs):
        
        self.mode = mode
        
        if mode == 'horizontal':
            self.horizontal = True
            self.vertical = False
        elif mode == 'vertical':
            self.horizontal = False
            self.vertical = True
        elif mode == 'horizontal_and_vertical':
            self.horizontal = True
            self.vertical = True
        
        super(RandomFlipF, self).__init__(name=name, **kwargs)

    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()

        def random_flip():
            outputs = tf.identity(inputs)
            if self.horizontal:
                outputs = tf.image.random_flip_left_right(inputs)
            if self.vertical:
                outputs = tf.image.random_flip_up_down(inputs)
            return outputs

        output = tf.cond(tf.equal(training, True), random_flip, lambda: inputs)
        output.set_shape(inputs.shape)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "mode": self.mode,
        })
        return config
    

# Speed up https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomCrop applying the transformation directly on the whole batch
class RandomCropF(tfkl.Layer):
    def __init__(self, height, width, padding=(0,0), fill_value=0.0, name=None, **kwargs):
        
        self.height = height
        self.width = width
        self.fill_value = fill_value
        
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, "__len__"):
            if len(padding) != 2:
                raise ValueError(f"`padding` should have two elements. Received: {padding}.")
            height_padding = keras.utils.conv_utils.normalize_tuple(
                padding[0], 2, "1st entry of padding", allow_zero=True
            )
            width_padding = keras.utils.conv_utils.normalize_tuple(
                padding[1], 2, "2nd entry of padding", allow_zero=True
            )
            self.padding = (height_padding, width_padding)
        
        self.pattern = [[0, 0], list(self.padding[0]), list(self.padding[1]), [0, 0]]
        
        super(RandomCropF, self).__init__(name=name, **kwargs)

    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()

        def random_crop():
            outputs = tf.identity(inputs)
            outputs = tf.pad(outputs, self.pattern, constant_values=self.fill_value)
            input_shape = tf.shape(outputs)
            b,h,w,c = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
            
            ij = tf.stack(tf.meshgrid(
                tf.range(b, dtype=tf.int32),
                tf.range(self.height, dtype=tf.int32), 
                tf.range(self.width, dtype=tf.int32),
                indexing='ij'), axis=-1)

            h_diff = h - self.height
            w_diff = w - self.width
            dtype = input_shape.dtype
            rands = tf.random.uniform((2, b, 1, 1, 1), minval=0, maxval=dtype.max, dtype=dtype)
            top = rands[0] % (h_diff + 1)
            left = rands[1] % (w_diff + 1)
            top_left = tf.concat([top, left], axis=-1)
            
            offset = tf.concat([tf.zeros((b,1,1,1), dtype=tf.int32), top_left], -1)
            offset = tf.broadcast_to(offset, tf.shape(ij))
            
            ij = ij + offset
            outputs = tf.gather_nd(outputs, ij)
            
            return outputs
        
        output = tf.cond(tf.equal(training, True), random_crop, lambda: inputs)
        output.set_shape(inputs.shape)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "height": self.height,
            "width": self.width,
            "fill_value": self.fill_value,
            "padding": self.padding,
        })
        return config
    

def augment(image_size, normalize, random_flip=True, random_crop=True, random_rotation=False, random_zoom=False, random_erasing=False, autocast=False):
    """
    Returns a tf.keras.Sequential model that applies a series of image augmentations to an input image.

    Args:
        image_size (int): The size of the output image.
        normalize (bool): Whether the input image is normalized.
        random_flip (bool): Whether to randomly flip the image horizontally.
        random_crop (bool): Whether to randomly crop the image.
        random_rotation (bool): Whether to randomly rotate the image.
        random_zoom (bool): Whether to randomly zoom the image.
        random_erasing (bool): Whether to randomly erase parts of the image.
        autocast (bool): Whether to use autocasting. If False, the input image is not casted to the compute dtype of the layer.
                         Can be useful when using mixed precision training, but still want to use the augmentation layer with float32 images.

    Returns:
        A tf.keras.Sequential model that applies a series of image augmentations to an input image.
    """
    fill_value = -1.0 if normalize else 0.0
    
    def _augment():

        layers = []

        if random_rotation:
            trsfm = RandomRotationF(factor_degree=10, interpolation='bilinear', fill_value=fill_value, name='random_rotation', autocast=autocast)
            layers.append(trsfm)

        if random_flip:
            trsfm = RandomFlipF(mode='horizontal', name='random_flip', autocast=autocast)
            layers.append(trsfm)

        if random_zoom:
            # random_zoom = tfkl.RandomZoom(height_factor=(-0.2, 0.2), fill_mode='constant', interpolation='nearest', autocast=autocast)
            # random_zoom.auto_vectorize = False
            # trsfm = random_zoom
            trsfm = RandomZoomF(height_factor=(-0.2, 0.2), fill_mode='constant', interpolation='nearest', fill_value=fill_value, name='random_zoom', autocast=autocast)
            layers.append(trsfm)

        if random_crop:
            # N.B: here 'fill_value' is set to 0, no matter if the image is normalized or not, only for legacy reasons
            # i.e.: the paper results were obtained with this configuration
            trsfm = RandomCropF(image_size, image_size, padding=4, fill_value=0.0, name='random_crop', autocast=autocast)
            layers.append(trsfm)
            
        if random_erasing:
            trsfm = keras_cv.layers.RandomCutout(height_factor=(0.2, 0.5), width_factor=(0.2, 0.5), fill_mode='gaussian_noise',
                                                 fill_value=fill_value, name='random_cutout', autocast=autocast)
            keras_cv.layers.RandomCutout._compute_rectangle_fill = _compute_rectangle_fill # replace original method with ours
            keras_cv.layers.BaseImageAugmentationLayer._ensure_inputs_are_compute_dtype = _ensure_inputs_are_compute_dtype # replace original method with ours
            layers.append(trsfm)

        augmentation_layers = tfk.Sequential(layers, name='augmentation_layer')

        return augmentation_layers
    
    return _augment()