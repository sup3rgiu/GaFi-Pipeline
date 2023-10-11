import tensorflow as tf
from typing import Any, Text, Tuple, Type, Union

tfkl = tf.keras.layers

def _l2normalize(v, eps=1e-12):
    return v / (tf.reduce_sum(v**2)**0.5 + eps)

def spectral_norm(weights, u, num_iters, training):
    w_shape = weights.shape.as_list()
    w_mat = tf.reshape(weights, [-1, w_shape[-1]])

    u_ = u
    for _ in range(num_iters):
        v_ = _l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
        u_ = _l2normalize(tf.matmul(v_, w_mat))

    sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
    w_mat /= sigma
    w_bar = tf.reshape(w_mat, w_shape)
    if training:
        u_ = tf.cast(u_, u.dtype)
        u.assign(u_)
    return w_bar

class SDense(tfkl.Layer):
    def __init__(self,
                 units,
                 use_bias=True,
                 is_sn=False,
                 sn_iters=1,
                 initializer=tf.initializers.orthogonal(),
                 name='dense',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.units = units
        self.use_bias = use_bias
        self.sn_iters = sn_iters
        self.is_sn = is_sn
        self.initializer = initializer

    def build(self, input_shape):
        in_features = int(input_shape[-1])
        kernel_shape = [in_features, self.units]
        self.kernel = self.add_weight('kernel', shape=kernel_shape,
                                      initializer=self.initializer)
        if self.use_bias:
            self.bias = self.add_weight('bias', shape=[self.units],
                                        initializer=tf.zeros_initializer())
        if self.is_sn:
            self.u = self.add_weight('u', shape=[1, self.units], trainable=False)

    def call(self, x, training=None):
        if self.is_sn:
            w_bar = spectral_norm(self.kernel, self.u, self.sn_iters, training)
        else:
            w_bar = self.kernel
        x = tf.matmul(x, w_bar)
        
        if self.use_bias:
            return x + self.bias
        else:
            return x
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "use_bias": self.use_bias,
            "sn_iters": self.sn_iters,
            "is_sn": self.is_sn,
        })
        return config

class Conv(tfkl.Layer):
    def __init__(self,
                 filters,
                 kernel_size=(3,3),
                 strides=(1, 1),
                 padding='valid',
                 use_bias=True,
                 is_sn=False,
                 sn_iters=1,
                 initializer=tf.initializers.orthogonal(),
                 name='conv2d',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        if isinstance(padding, str):
            self._padding = padding.upper()
        elif isinstance(padding, int):
            self._padding = [[0, 0], [padding, padding],[padding, padding], [0, 0]]
        else:
            raise ValueError("Padding is in an invalid format. It must be either 'same', 'valid' or an integer.")
        self.use_bias = use_bias
        self.sn_iters = sn_iters
        self.is_sn = is_sn
        self.initializer = initializer

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        kernel_shape = [*self.kernel_size, in_channels, self.filters]
        self.kernel = self.add_weight('kernel', shape=kernel_shape,
                                      initializer=self.initializer)
        if self.use_bias:
            self.bias = self.add_weight('bias', shape=[self.filters],
                                        initializer=tf.zeros_initializer())
        if self.is_sn:
            self.u = self.add_weight('u', shape=[1, self.filters], trainable=False)

    def call(self, x, training=None):
        if self.is_sn:
            w_bar = spectral_norm(self.kernel, self.u, self.sn_iters, training)
        else:
            w_bar = self.kernel
        x = tf.nn.conv2d(x, w_bar, strides=[1, *self.strides, 1], padding=self._padding)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "use_bias": self.use_bias,
            "sn_iters": self.sn_iters,
            "is_sn": self.is_sn,
        })
        return config

class ConditionalBatchNorm(tfkl.Layer):
    def __init__(self,
                 num_features,
                 momentum=0.9,
                 epsilon=1E-5,
                 initializer=tf.initializers.orthogonal(),
                 is_sn=True,
                 name='conditional_batch_norm',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon
        self.is_sn = is_sn
        self.initializer = initializer

    def build(self, input_shape):
        self.beta = SDense(self.num_features, use_bias=False, is_sn=self.is_sn, initializer=self.initializer, name='linear_beta')
        self.gamma = SDense(self.num_features, use_bias=False, is_sn=self.is_sn, initializer=self.initializer, name='linear_gamma')
        self.bn = tfkl.BatchNormalization(momentum=self.momentum, epsilon=self.epsilon, center=False, scale=False)
        
    @tf.function(input_signature=[]) # needed to save and reload this method
    def reset_standing_stats(self):
        self.bn.moving_mean.assign(tf.zeros_like(self.bn.moving_mean))
        self.bn.moving_variance.assign(tf.ones_like(self.bn.moving_variance))
        
    def call(self, inputs, training=None):
        x, condition = inputs
                
        condition = tf.cast(condition, tf.float32) # needed when condition is an integer, otherwise we can't pass it in Dense layer
                
        beta = self.beta(condition, training=training) # bias
        gamma = 1 + self.gamma(condition, training=training) # gain
        
        beta = tf.expand_dims(tf.expand_dims(beta, 1), 1)
        gamma = tf.expand_dims(tf.expand_dims(gamma, 1), 1)
        
        out = self.bn(x, training=training)
        
        return out * gamma + beta
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_features": self.num_features,
            "momentum": self.momentum,
            "epsilon": self.epsilon,
            "is_sn": self.is_sn,
        })
        return config

class SelfAttention(tfkl.Layer):
    def __init__(self,
                 initializer=tf.initializers.orthogonal(),
                 name='self_attention',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.initializer = initializer

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        self.conv_theta = Conv(filters=in_channels//8,
                               kernel_size=(1,1),
                               strides=(1,1),
                               is_sn=True,
                               use_bias=False,
                               initializer=self.initializer,
                               name='sn_conv_theta')
        self.conv_phi = Conv(filters=in_channels//8,
                               kernel_size=(1,1),
                               strides=(1,1),
                               is_sn=True,
                               use_bias=False,
                               initializer=self.initializer,
                               name='sn_conv_phi')
        self.conv_g = Conv(filters=in_channels//2,
                               kernel_size=(1,1),
                               strides=(1,1),
                               is_sn=True,
                               use_bias=False,
                               initializer=self.initializer,
                               name='sn_conv_g')
        self.conv_attn = Conv(filters=in_channels,
                               kernel_size=(1,1),
                               strides=(1,1),
                               is_sn=True,
                               use_bias=False,
                               initializer=self.initializer,
                               name='sn_conv_attn')
        self.sigma = self.add_weight('sigma', shape=[],
                                     initializer=tf.zeros_initializer())

    def call(self, x, training=None):
        x_shape = tf.shape(x)
        batch_size, h, w, in_channels = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        location_num = h*w
        downsampled_num = location_num//4

        theta = self.conv_theta(x, training=training)
        theta = tf.reshape(theta, [batch_size, location_num, in_channels//8])

        phi = self.conv_phi(x, training=training)
        phi = tf.nn.max_pool(phi, ksize=[2, 2], strides=2, padding='VALID')
        phi = tf.reshape(phi, [batch_size, downsampled_num, in_channels//8])

        attn = tf.matmul(theta, phi, transpose_b=True)
        attn = tf.nn.softmax(attn)

        g = self.conv_g(x, training=training)
        g = tf.nn.max_pool(g, ksize=[2, 2], strides=2, padding='VALID')
        g = tf.reshape(g, [batch_size, downsampled_num, in_channels//2])

        attn_g = tf.matmul(attn, g)
        attn_g = tf.reshape(attn_g, [batch_size, h, w, in_channels//2])
        attn_g = self.conv_attn(attn_g, training=training)

        # tf.cast needed to be able to reload a full-precision saved model when using mixed precision (hacky...)
        return x + tf.cast(self.sigma, x.dtype) * tf.cast(attn_g, x.dtype)
    
    def get_config(self):
        config = super().get_config()
        config.update({})
        return config

class SEmbedding(tfkl.Layer):
    def __init__(self,
                 num_classes,
                 embedding_size,
                 is_sn=True,
                 sn_iters=1,
                 initializer=tf.initializers.orthogonal(),
                 name='embedding',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.sn_iters = sn_iters
        self.is_sn = is_sn
        self.initializer = initializer

    def build(self, input_shape):
        embed_shape = [self.num_classes, self.embedding_size]
        self.embed_map = self.add_weight('embed_map', shape=embed_shape,
                                         initializer=self.initializer)
        if self.is_sn:
            self.u = self.add_weight('u', shape=[1, self.num_classes], trainable=False)

    def call(self, x, training=None):
        dtype = tf.keras.backend.dtype(x)
        if dtype != "int32" and dtype != "int64":
            x = tf.cast(x, "int32")
        
        if self.is_sn:
            embed_map_bar_T = tf.transpose(self.embed_map)
            embed_map_bar_T = spectral_norm(embed_map_bar_T, self.u, self.sn_iters, training)
            embed_map_bar = tf.transpose(embed_map_bar_T)
        else:
            embed_map_bar =  self.embed_map
            
        x = tf.nn.embedding_lookup(embed_map_bar, x)
        
        # The output shape is (batch_size, 1, output_dim). We don't want the 1
        x = tf.squeeze(x, axis=1)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "embedding_size": self.embedding_size,
            "sn_iters": self.sn_iters,
            "is_sn": self.is_sn,
        })
        return config

class BlurPool2D(tf.keras.layers.Layer):
    """A layer to do channel-wise blurring + subsampling on 2D inputs.

    Reference:
    Zhang et al. Making Convolutional Networks Shift-Invariant Again.
    https://arxiv.org/pdf/1904.11486.pdf.
    """

    def __init__(self,
               filter_size: int = 3,
               strides: Tuple[int, int, int, int] = (1, 2, 2, 1),
               padding: Text = "SAME",
               **kwargs: Any) -> None:
        """Initializes the BlurPool2D layer.

        Args:
          filter_size: The size (height and width) of the blurring filter.
          strides: The stride for convolution of the blurring filter for each
            dimension of the inputs.
          padding: One of 'VALID' or 'SAME', specifying the padding type used for
            convolution.
          **kwargs: Keyword arguments forwarded to super().__init__().

        Raises:
          ValueError: If filter_size is not 3, 4, 5, 6 or 7.
        """
        if filter_size not in [3, 4, 5, 6, 7]:
            raise ValueError("Only filter_size of 3, 4, 5, 6 or 7 is supported.")
        super().__init__(**kwargs)
        self._filter_size = filter_size
        self._strides = strides
        self._padding = padding

        if filter_size == 3:
            self._filter = [1., 2., 1.]
        elif filter_size == 4:
            self._filter = [1., 3., 3., 1.]
        elif filter_size == 5:
            self._filter = [1., 4., 6., 4., 1.]
        elif filter_size == 6:
            self._filter = [1., 5., 10., 10., 5., 1.]
        elif filter_size == 7:
            self._filter = [1., 6., 15., 20., 15., 6., 1.]

        self._filter = tf.constant(self._filter, dtype=tf.float32)
        self._filter = self._filter[:, None] * self._filter[None, :]
        self._filter /= tf.reduce_sum(self._filter)
        self._filter = tf.reshape(self._filter, [self._filter.shape[0], self._filter.shape[1], 1, 1])

    def build(self, input_shape: tf.TensorShape) -> None:
        self._filter = tf.tile(self._filter, [1, 1, input_shape[-1], 1])
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Calls the BlurPool2D layer on the given inputs."""
        return tf.nn.depthwise_conv2d(
            input=inputs,
            filter=self._filter,
            strides=self._strides,
            padding=self._padding)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filter_size": self._filter_size,
            "strides": self._strides,
            "padding": self._padding,
        })
        return config

class PaddedConv2D(tfkl.Conv2D):
    def __init__(self, filters, kernel_size, padding=0, strides=1, name=None, **kwargs):
        self._padding = padding
        super().__init__(filters=filters, kernel_size=kernel_size, strides=strides, padding='valid', name=name, **kwargs)
        self.padding2d = tfkl.ZeroPadding2D(self._padding)

    def call(self, inputs):
        x = self.padding2d(inputs)
        return super().call(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "padding": self._padding,
        })
        return config

def GlobalSumPooling2D(name=None):
    """
    Layer that sums over all spatial locations,
    preserving batch and channels dimensions.
    """
    def call(x):
        return tf.reduce_sum(x, axis=(1, 2))

    def output_shape(input_shape):
        return input_shape[0], input_shape[-1]

    return tfkl.Lambda(call, output_shape=output_shape, name=name)

