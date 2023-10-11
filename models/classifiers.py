import tensorflow as tf
from tensorflow.keras import backend as K
import keras
from functools import partial
from models.layers import PaddedConv2D

tfk = tf.keras
tfkl = tf.keras.layers

############# Simple Classifier #############

def Simple_Classifier(
    input_shape,
    num_classes,
    seed=None,
    augmentation_layer=None,
):
    
    tf.random.set_seed(seed)

    # Input image
    image = tfkl.Input(input_shape, name='image')
    h = image
    
    if augmentation_layer is not None:
        h = augmentation_layer(h)
        
    init = partial(tfk.initializers.HeNormal, seed=seed)
    
    h = tfkl.Conv2D(filters=256, kernel_size=2, padding='same', activation='relu', kernel_initializer=init(), name=f"conv2d_1")(h)
    h = tfkl.MaxPooling2D(pool_size=2, name='pool2d_1')(h)
    h = tfkl.Dropout(0.3, name='dropout_1')(h)
    
    h = tfkl.Conv2D(filters=128, kernel_size=2, padding='same', activation='relu', kernel_initializer=init(), name=f"conv2d_2")(h)
    h = tfkl.MaxPooling2D(pool_size=2, name='pool2d_2')(h)
    h = tfkl.Dropout(0.3, name='dropout_2')(h)
    
    h = tfkl.Flatten(name='flatten')(h)
    h = tfkl.Dense(256, activation='relu', kernel_initializer=init(), name='dense_1')(h)
    h = tfkl.Dropout(0.3, name='dropout_3')(h)
    h = tfkl.Dense(num_classes, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed), name='dense_logits')(h)
    h = tfkl.Activation('softmax', dtype='float32', name='out')(h)
    
    return tfk.Model(image, h, name="Simple_Classifier")

############# ResNet StudioGAN #############

def BasicBlock(x, out_channels, init, stride=1, downsample=False, **kwargs):
    expansion = 1
   
    input_dim = K.int_shape(x)[-1]
    i = kwargs['i']
    temp = tf.identity(x)
    
    h = PaddedConv2D(filters=out_channels, kernel_size=3, strides=stride, padding=1, use_bias=False, kernel_initializer=init(), name=f"conv_1_{i}")(x)
    h = tfkl.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"bn_1_{i}")(h)
    h = tfkl.ReLU(name=f"relu_1_{i}")(h)
    
    h = PaddedConv2D(filters=out_channels, kernel_size=3, strides=stride, padding=1, use_bias=False, kernel_initializer=init(), name=f"conv_2_{i}")(h)
    h = tfkl.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"bn_2_{i}")(h)
    
    if downsample:
        temp = tfkl.Conv2D(filters=out_channels*expansion, kernel_size=1, strides=stride, padding='valid', use_bias=False, kernel_initializer=init(), name=f"down_conv_{i}")(temp)
        temp = tfkl.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"down_bn_{i}")(temp)
    
    h = tfkl.Add(name=f"add_{i}")([h, temp])
    h = tfkl.ReLU(name=f"relu_2_{i}")(h)

    return h

def Bottleneck(x, out_channels, init, stride=1, downsample=False, **kwargs):
    expansion = 4
   
    input_dim = K.int_shape(x)[-1]
    i = kwargs['i']
    temp = tf.identity(x)
    
    h = tfkl.Conv2D(filters=out_channels, kernel_size=1, strides=1, padding='valid', use_bias=False, kernel_initializer=init(), name=f"conv_1_{i}")(x)
    h = tfkl.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"bn_1_{i}")(h)
    h = tfkl.ReLU(name=f"relu_1_{i}")(h)
    
    h = PaddedConv2D(filters=out_channels, kernel_size=3, strides=stride, padding=1, use_bias=False, kernel_initializer=init(), name=f"conv_2_{i}")(h)
    h = tfkl.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"bn_2_{i}")(h)
    h = tfkl.ReLU(name=f"relu_2_{i}")(h)
    
    h = tfkl.Conv2D(filters=out_channels*expansion, kernel_size=1, strides=1, padding='valid', use_bias=False, kernel_initializer=init(), name=f"conv_3_{i}")(h)
    h = tfkl.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"bn_3_{i}")(h)
    
    if downsample:
        temp = tfkl.Conv2D(filters=out_channels*expansion, kernel_size=1, strides=stride, padding='valid', use_bias=False, kernel_initializer=init(), name=f"down_conv_{i}")(temp)
        temp = tfkl.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"down_bn_{i}")(temp)
    
    h = tfkl.Add(name=f"add_{i}")([h, temp])
    h = tfkl.ReLU(name=f"relu_3_{i}")(h)

    return h

def ResNet_StudioGAN(
    input_shape,
    num_classes,
    depth,
    bottleneck=False,
    seed=None,
    augmentation_layer=None,
):
    
    tf.random.set_seed(seed)
    
    inplanes = 16
    if bottleneck == True:
        expansion = 4
        n = int((depth - 2) / 9)
        block = Bottleneck
    else:
        expansion = 1
        n = int((depth - 2) / 6)
        block = BasicBlock
        
    def _make_layer(h, block, planes, blocks, init, i, stride=1):
        downsample = False
        if stride != 1 or inplanes != planes * expansion:
            downsample = True
        h = block(x=h, out_channels=planes, stride=stride, downsample=downsample, init=init, i=f'{i}_0')
        for j in range(1, blocks):
            h = block(x=h, out_channels=planes, init=init, i=f'{i}_{j}')
        return h
    
    # Input image
    image = tfkl.Input(input_shape, name='image')
    h = image
    
    if augmentation_layer is not None:
        h = augmentation_layer(h)
        
    init = partial(tfk.initializers.HeNormal, seed=seed)
    
    h = PaddedConv2D(filters=inplanes, kernel_size=3, strides=1, padding=1, use_bias=False, kernel_initializer=init(), name=f"conv_init")(h)
    h = tfkl.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"bn_init")(h)
    h = tfkl.ReLU(name=f"relu_init")(h)
    
    # Downsacale ResNet blocks
    h = _make_layer(h, block, 16, n, init=init, i=1)
    h = _make_layer(h, block, 32, n, init=init, i=2, stride=2)
    h = _make_layer(h, block, 64, n, init=init, i=3, stride=2)

    # Network head
    h = tfkl.AveragePooling2D(pool_size=8, name='average_pooling')(h)
    h = tfkl.Flatten(name='flatten')(h)
    h = tfkl.Dense(num_classes, name='dense_logits')(h)
    h = tfkl.Activation('softmax', dtype='float32', name='out')(h)
    
    # Return Keras model
    return tfk.Model(image, h, name="ResNet_StudioGAN")

############# ResNet #############

# Modified from https://github.com/HabibSlim/ClassifierTrainingGAN/blob/master/classifiers/resnetw.py

class CustomModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # We use a weight decay of 0.0002, which performs better
        # than the 0.0001 that was originally suggested.
        self.weight_decay = 2e-4
    
    def train_step(self, data):
        x, y, sample_weight = keras.engine.data_adapter.unpack_x_y_sample_weight(data)
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight)
            loss += self.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables])
        self._validate_target_and_loss(y, loss)
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(x, y, y_pred, sample_weight)

def BasicBlockW(x, planes, init, stride=1, option='A', **kwargs):
    expansion = 1
   
    input_dim = K.int_shape(x)[-1]
    i = kwargs['i']
    temp = tf.identity(x)
    
    h = PaddedConv2D(filters=planes, kernel_size=3, strides=stride, padding=1, use_bias=False, kernel_initializer=init(), name=f"conv_1_{i}")(x)
    h = tfkl.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"bn_1_{i}")(h)
    h = tfkl.ReLU(name=f"relu_1_{i}")(h)
    
    h = PaddedConv2D(filters=planes, kernel_size=3, strides=1, padding=1, use_bias=False, kernel_initializer=init(), name=f"conv_2_{i}")(h)
    h = tfkl.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"bn_2_{i}")(h)
    
    if stride != 1 or input_dim != planes:
        if option == 'A':
            """
            For CIFAR10 ResNet paper uses option A.
            """
            temp = tf.pad(temp[:,::2,::2,:], [[0,0],[0,0],[0,0], [planes//4,planes//4]], mode='CONSTANT', constant_values=0)
        elif option == 'B':
            temp = tfkl.Conv2D(filters=planes*expansion, kernel_size=1, strides=stride, padding='valid', use_bias=False, kernel_initializer=init(), name=f"down_conv_{i}")(temp)
            temp = tfkl.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"down_bn_{i}")(temp)
    
    h = tfkl.Add(name=f"add_{i}")([h, temp])
    h = tfkl.ReLU(name=f"relu_2_{i}")(h)

    return h

def ResNetW(
        input_shape,
        block,
        num_blocks,
        num_classes,
        width=16,
        use_w_decay=False,
        seed=None,
        augmentation_layer=None,
        name=''):
    
    tf.random.set_seed(seed)

    x0 = tfkl.Input(shape=input_shape, name='input_image')
    inputs = x0
    
    if augmentation_layer is not None:
        inputs = augmentation_layer(inputs)

    init = partial(tfk.initializers.HeNormal, seed=seed)
    
    inputs = PaddedConv2D(filters=width, kernel_size=3, strides=1, padding=1, use_bias=False, kernel_initializer=init(), name='initial_conv')(inputs)
    inputs = tfkl.BatchNormalization(momentum=0.9, epsilon=1e-5, name='initial_bn')(inputs)
    inputs = tfkl.ReLU(name='initial_relu')(inputs)

    j = 0
    def _make_layer(inputs, block, planes, num_blocks, stride):
        nonlocal j
        strides = [stride] + [1] * (num_blocks - 1)
        for i, stride in enumerate(strides):
            inputs = block(inputs, planes, init, stride=stride, option='A', i=f'{j}_{i}')
        j += 1
        return inputs
    
    inputs = _make_layer(inputs, block, width, num_blocks[0], stride=1)
    inputs = _make_layer(inputs, block, width * 2, num_blocks[1], stride=2)
    inputs = _make_layer(inputs, block, width * 4, num_blocks[2], stride=2)
    
    inputs = tfkl.AveragePooling2D(pool_size=inputs.shape[1], name='avg_pool2d')(inputs)
    inputs = tfkl.Flatten(name='flatten')(inputs)
    outputs = tfkl.Dense(units=num_classes, name='dense_logits', kernel_initializer=init())(inputs)
    outputs = tfkl.Activation('softmax', name='out', dtype='float32')(outputs)
    
    if use_w_decay:
        return CustomModel(x0, outputs, name=name+'_w')
    else:
        return tfk.Model(x0, outputs, name=name)
    
def resnet14(input_shape, num_classes, width=16, seed=None, augmentation_layer=None):
    return ResNetW(input_shape, BasicBlockW, [2, 2, 2], num_classes, width, seed=seed, name='resnet14', augmentation_layer=augmentation_layer)

def resnet20(input_shape, num_classes, width=16, seed=None, augmentation_layer=None):
    return ResNetW(input_shape, BasicBlockW, [3, 3, 3], num_classes, width, seed=seed, name='resnet20', augmentation_layer=augmentation_layer)

def resnet32(input_shape, num_classes, width=16, seed=None, augmentation_layer=None):
    return ResNetW(input_shape, BasicBlockW, [5, 5, 5], num_classes, width, seed=seed, name='resnet32', augmentation_layer=augmentation_layer)

def resnet44(input_shape, num_classes, width=16, seed=None, augmentation_layer=None):
    return ResNetW(input_shape, BasicBlockW, [7, 7, 7], num_classes, width, seed=seed, name='resnet44', augmentation_layer=augmentation_layer)

def resnet56(input_shape, num_classes, width=16, seed=None, augmentation_layer=None):
    return ResNetW(input_shape, BasicBlockW, [9, 9, 9], num_classes, width, seed=seed, name='resnet56', augmentation_layer=augmentation_layer)

def resnet110(input_shape, num_classes, width=16, seed=None, augmentation_layer=None):
    return ResNetW(input_shape, BasicBlockW, [18, 18, 18], num_classes, width, seed=seed, name='resnet110', augmentation_layer=augmentation_layer)

def resnet1202(input_shape, num_classes, width=16, seed=None, augmentation_layer=None):
    return ResNetW(input_shape, BasicBlockW, [200, 200, 200], num_classes, width, seed=seed, name='resnet1202', augmentation_layer=augmentation_layer)

############# Utils #############

def get_classifier(model, input_shape, num_classes, width=None, seed=None, augmentation_layer=None):
    if model == 'resnet14':
        return resnet14(input_shape, num_classes, width, seed=seed, augmentation_layer=augmentation_layer)
    elif model == 'resnet20':
        return resnet20(input_shape, num_classes, width, seed=seed, augmentation_layer=augmentation_layer)
    elif model == 'resnet32':
        return resnet32(input_shape, num_classes, width, seed=seed, augmentation_layer=augmentation_layer)
    elif model == 'resnet44':
        return resnet44(input_shape, num_classes, width, seed=seed, augmentation_layer=augmentation_layer)
    elif model == 'resnet56':
        return resnet56(input_shape, num_classes, width, seed=seed, augmentation_layer=augmentation_layer)
    elif model == 'resnet110':
        return resnet110(input_shape, num_classes, width, seed=seed, augmentation_layer=augmentation_layer)
    elif model == 'resnet1202':
        return resnet1202(input_shape, num_classes, width, seed=seed, augmentation_layer=augmentation_layer)
    elif model == 'resnet_studiogan':
        return ResNet_StudioGAN(input_shape, num_classes, width, seed=seed, augmentation_layer=augmentation_layer)
    elif model == 'simple':
        return Simple_Classifier(input_shape, num_classes, seed=seed, augmentation_layer=augmentation_layer)
    else:
        raise NotImplementedError(f"Model {model} currently not implemented")