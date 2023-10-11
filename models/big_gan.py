import tensorflow as tf
from tensorflow.keras import backend as K
from models.layers import Conv, ConditionalBatchNorm, BlurPool2D, SEmbedding, SDense, SelfAttention, GlobalSumPooling2D
from functools import partial

tfk = tf.keras
tfkl = tf.keras.layers

############# Blocks #############

def G_Resblock(
    x: tf.Tensor,
    y: tf.Tensor,
    *,
    out_channels: int,
    upsample: bool,
    concat: bool = False,
    momentum: float = 0.9,
    epsilon: float = 1E-5,
    **kwargs
):

    inputs = [x, y]
    
    input_dim = K.int_shape(x)[-1]
    i = kwargs['i']
    temp = tf.identity(x)
    
    learnable_sc = input_dim != out_channels or upsample
    
    h = ConditionalBatchNorm(num_features=input_dim, is_sn=True, momentum=momentum, epsilon=epsilon, name=f"ccbn_1_{i}")(inputs)
    h = tfkl.ReLU(name=f"act_1_{i}")(h)
    h = tfkl.UpSampling2D(name=f"upsample_h_{i}")(h)
    h = Conv(filters=out_channels, kernel_size=(3,3), strides=(1, 1), padding=1, is_sn=True, name=f"conv_1_{i}")(h)
    h = ConditionalBatchNorm(num_features=out_channels, is_sn=True, momentum=momentum, epsilon=epsilon, name=f"ccbn_2_{i}")([h, y])
    h = tfkl.ReLU(name=f"act_2_{i}")(h)
    h = Conv(filters=out_channels, kernel_size=(3,3), strides=(1, 1), padding=1, is_sn=True, name=f"conv_2_{i}")(h)
    
    #Identity mapping
    temp = tfkl.UpSampling2D(name=f"upsample_x_{i}")(temp)
    if learnable_sc:
        temp = Conv(filters=out_channels, kernel_size=(1,1), strides=(1, 1), padding=0, is_sn=True, name=f"conv_identity_{i}")(temp)
    
    if concat:
        out = tfkl.Concatenate(name=f"concat_{i}")([h, temp])
        out = Conv(filters=out_channels, kernel_size=(1,1), strides=(1, 1), padding=0, is_sn=True, name=f"res_block_up_output_{i}")(out)
    else:
        out = tfkl.Add(name=f"res_block_up_output_{i}")([h, temp])
        
    return out

def G_Resblock_Deep(
    x: tf.Tensor,
    y: tf.Tensor,
    *,
    out_channels: int,
    upsample: bool = True,
    studiogan: bool = True,
    concat: bool = False,
    channel_ratio: int = 4,
    momentum: float = 0.9,
    epsilon: float = 1E-5,
    **kwargs
):
    
    in_channels = K.int_shape(x)[-1]
    hidden_channels = in_channels // channel_ratio
    
    i = kwargs['i']
    
    inputs = [x, y]
    temp = tf.identity(x)
    # Drop channels in x if necessary
    if in_channels != out_channels and not studiogan:
        temp = temp[..., :out_channels]
    
    # Project down to channel ratio
    h = ConditionalBatchNorm(num_features=in_channels, is_sn=True, momentum=momentum, epsilon=epsilon, name=f"ccbn_1_{i}")(inputs)
    h = tfkl.ReLU(name=f"act_1_{i}")(h)
    h = Conv(filters=hidden_channels, kernel_size=(1,1), strides=(1, 1), padding=0, is_sn=True, name=f"conv_1_{i}")(h)
    
    # Apply next BN-ReLU
    h = ConditionalBatchNorm(num_features=hidden_channels, is_sn=True, momentum=momentum, epsilon=epsilon, name=f"ccbn_2_{i}")([h, y])
    h = tfkl.ReLU(name=f"act_2_{i}")(h)
            
    # Upsample both h and x at this point
    if upsample:
        h = tfkl.UpSampling2D(name=f"upsample_h_{i}")(h)
        temp = tfkl.UpSampling2D(name=f"upsample_x_{i}")(temp)
    
    # 3x3 convs
    h = Conv(filters=hidden_channels, kernel_size=(3,3), strides=(1, 1), padding=1, is_sn=True, name=f"conv_2_{i}")(h)
    h = ConditionalBatchNorm(num_features=hidden_channels, is_sn=True, momentum=momentum, epsilon=epsilon, name=f"ccbn_3_{i}")([h, y])
    h = tfkl.ReLU(name=f"act_3_{i}")(h)
    h = Conv(filters=hidden_channels, kernel_size=(3,3), strides=(1, 1), padding=1, is_sn=True, name=f"conv_3_{i}")(h)
    
    # Final 1x1 conv
    h = ConditionalBatchNorm(num_features=hidden_channels, is_sn=True, momentum=momentum, epsilon=epsilon, name=f"ccbn_4_{i}")([h, y])
    h = tfkl.ReLU(name=f"act_4_{i}")(h)
    h = Conv(filters=out_channels, kernel_size=(1,1), strides=(1, 1), padding=0, is_sn=True, name=f"conv_4_{i}")(h)
    
    if studiogan:
        temp = Conv(filters=out_channels, kernel_size=(1,1), strides=(1, 1), padding=0, is_sn=True, name=f"conv_identity_{i}")(temp)
        
    if concat:
        out = tfkl.Concatenate(name=f"concat_{i}")([h, temp])
        out = Conv(filters=out_channels, kernel_size=(1,1), strides=(1, 1), padding=0, is_sn=True, name=f"res_block_{'up_' if upsample else ''}output_{i}")(out)
    else:
        out = tfkl.Add(name=f"res_block_{'up_' if upsample else ''}output_{i}")([h, temp])

    return out

def D_Resblock(
    x: tf.Tensor,
    *,
    out_channels: int,
    wide: bool = True,
    is_down: bool = True,
    is_first: bool = False,
    blur_resample: bool = False,
    concat: bool = False,
    **kwargs
):
    
    in_channels = K.int_shape(x)[-1]
    i = kwargs['i']
    temp = tf.identity(x)
    learnable_sc = in_channels != out_channels or is_down
    
    hidden_channels = out_channels if wide else in_channels
    
    if blur_resample:
        downsample_layer = partial(BlurPool2D, filter_size=4)
    else:
        downsample_layer = partial(tfkl.AveragePooling2D)
    
    if is_first:
        h = x
    else:
        h = tfkl.ReLU(name=f"act_1_{i}")(x) 
    h = Conv(filters=hidden_channels, kernel_size=(3,3), strides=(1, 1), padding=1, is_sn=True, name=f"conv_1_{i}")(h)
    h = tfkl.ReLU(name=f"act_2_{i}")(h)
    h = Conv(filters=out_channels, kernel_size=(3,3), strides=(1, 1), padding=1, is_sn=True, name=f"conv_2_{i}")(h)
    
    if is_down:
        h = downsample_layer(name=f"downsample_h_{i}")(h)
        
    #Identity mapping
    if is_first:
        if is_down:
            temp = downsample_layer(name=f"downsample_x_{i}")(temp)
        if learnable_sc:
            temp = Conv(filters=out_channels, kernel_size=(1,1), strides=(1, 1), padding=0, is_sn=True, name=f"conv_identity_{i}")(temp)
    else:
        if learnable_sc:
            temp = Conv(filters=out_channels, kernel_size=(1,1), strides=(1, 1), padding=0, is_sn=True, name=f"conv_identity_{i}")(temp)
        if is_down:
            temp = downsample_layer(name=f"downsample_x_{i}")(temp)
            
    if concat:
        out = tfkl.Concatenate(name=f"concat_{i}")([h, temp])
        out = Conv(filters=out_channels, kernel_size=(1,1), strides=(1, 1), padding=0, is_sn=True, name=f"res_block_{'down_' if is_down else ''}output_{i}")(out)
    else:
        out = tfkl.Add(name=f"res_block_{'down_' if is_down else ''}output_{i}")([h, temp])
        
    return out

def D_Resblock_Deep(
    x: tf.Tensor,
    *,
    out_channels: int,
    is_first: bool = False,
    downsample: bool = True,
    studiogan: bool = True,
    concat: bool = False,
    channel_ratio: int = 4,
    blur_resample: bool = False,
    **kwargs
):
    
    in_channels = K.int_shape(x)[-1]
    hidden_channels = out_channels // channel_ratio
    learnable_sc = True if (in_channels != out_channels) else False
    
    i = kwargs['i']
    
    temp = tf.identity(x)
    
    if blur_resample:
        downsample_layer = partial(BlurPool2D, filter_size=4)
    else:
        downsample_layer = partial(tfkl.AveragePooling2D)
    
    # 1x1 bottleneck conv
    h = tfkl.ReLU(name=f"act_1_{i}")(x)
    h = Conv(filters=hidden_channels, kernel_size=(1,1), strides=(1, 1), padding=0, is_sn=True, name=f"conv_1_{i}")(h)
    
    # 3x3 convs
    h = tfkl.ReLU(name=f"act_2_{i}")(h)
    h = Conv(filters=hidden_channels, kernel_size=(3,3), strides=(1, 1), padding=1, is_sn=True, name=f"conv_2_{i}")(h)
    h = tfkl.ReLU(name=f"act_3_{i}")(h)
    h = Conv(filters=hidden_channels, kernel_size=(3,3), strides=(1, 1), padding=1, is_sn=True, name=f"conv_3_{i}")(h)
    
    if studiogan:
        if downsample:
            h = downsample_layer(name=f"downsample_h_{i}")(h)

        # final 1x1 conv
        h = tfkl.ReLU(name=f"act_4_{i}")(h)
        h = Conv(filters=out_channels, kernel_size=(1,1), strides=(1, 1), padding=0, is_sn=True, name=f"conv_4_{i}")(h)

        if is_first:
            temp = downsample_layer(name=f"downsample_x_{i}")(temp)
            temp = Conv(filters=out_channels, kernel_size=(1,1), strides=(1, 1), padding=0, is_sn=True, name=f"conv_identity_{i}")(temp)
        else:
            if downsample or learnable_sc:
                temp = Conv(filters=out_channels, kernel_size=(1,1), strides=(1, 1), padding=0, is_sn=True, name=f"conv_identity_{i}")(temp)
                if downsample:
                   temp = downsample_layer(name=f"downsample_x_{i}")(temp) 
    else:
        # legacy / original BigGAN Deep block

        # relu before downsample
        h = tfkl.ReLU(name=f"act_4_{i}")(h)
        
        if downsample:
            h = downsample_layer(name=f"downsample_h_{i}")(h)
            temp = downsample_layer(name=f"downsample_x_{i}")(temp)

            # final 1x1 conv
        h = Conv(filters=out_channels, kernel_size=(1,1), strides=(1, 1), padding=0, is_sn=True, name=f"conv_4_{i}")(h)
            
        if learnable_sc:
            extra = out_channels - in_channels
            temp_extra = Conv(filters=extra, kernel_size=(1,1), strides=(1, 1), padding=0, is_sn=True, name=f"conv_extra_{i}")(temp)
            temp = tfkl.Concatenate(name=f"concat_extra_{i}")([temp, temp_extra])

    if concat:
        out = tfkl.Concatenate(name=f"concat_{i}")([h, temp])
        out = Conv(filters=out_channels, kernel_size=(1,1), strides=(1, 1), padding=0, is_sn=True, name=f"res_block_{'down_' if downsample else ''}output_{i}")(out)
    else:
        out = tfkl.Add(name=f"res_block_{'down_' if downsample else ''}output_{i}")([h, temp])
        
    return out

############# Networks #############

def Generator(
    latent_dim,
    num_classes,
    hier = True,
    apply_attn = False,
    bn_epsilon = 1e-5,
    momentum = 0.9,
    out_channels = 1,
    output_activation = 'tanh'
):
    # Input z-vector.
    _z = tfkl.Input((latent_dim,), name='latent')
    
    # Input class label.
    _y = tfkl.Input((1,), dtype=tf.int32, name='class_label')
    y = SEmbedding(num_classes, latent_dim, is_sn=False, name='shared')(_y) # normal tfkl.Embedding, not SN (is_sn=False)
    
    if hier:
        num_slots = 4
        z_chunk_size = latent_dim // num_slots
        latent_dim = z_chunk_size * num_slots
        zs = tf.split(_z, num_slots, axis=1)
        z = zs[0]
        ys = [tf.concat([y, item], axis=1) for item in zs[1:]]
    else:
        ys = [y] * 3
        
    # Input class label.
    y = tfkl.Input((1,), dtype=tf.int32, name='class_label')
    
    # First linear layer + Reshape
    h = SDense(units=256*4*4, is_sn=True, name='first_linear')(z)
    h = tfkl.Reshape((4, 4, -1), name='first_reshape')(h)
    
    # ResBlocks + Attention
    h = G_Resblock(h, ys[0], out_channels=256, momentum=momentum, bn_epsilon=bn_epsilon, upsample=True, i=1)
    h = G_Resblock(h, ys[1], out_channels=256, momentum=momentum, bn_epsilon=bn_epsilon, upsample=True, i=2)
    if apply_attn:
        h = SelfAttention(name='non_local')(h)
    h = G_Resblock(h, ys[2], out_channels=256, momentum=momentum, bn_epsilon=bn_epsilon, upsample=True, i=3)
    
    # output layer: batchnorm-relu-conv.
    h = tfkl.BatchNormalization(epsilon=bn_epsilon, momentum=momentum, name='last_bn')(h)
    h = tfkl.ReLU(name='last_act')(h)
    h = Conv(filters=out_channels, kernel_size=(3,3), strides=(1, 1), padding=1, is_sn=True, name='last_conv')(h)
    
    # Output activation
    h = tfkl.Activation(output_activation, dtype="float32", name='output_activation')(h)

    # Return Keras model.
    return tfk.Model([_z, _y], h, name="Generator")

def GeneratorDeep(
    latent_dim,
    shared_dim,
    num_classes,
    img_size = 32,
    hier = True,
    g_conv_dim = 128,
    g_depth = 2,
    apply_attn = True,
    studiogan = True,
    concat = False,
    bn_epsilon = 1e-5,
    momentum = 0.9,
    out_channels = 1,
    output_activation = 'tanh'
):
    
    #### Configs ####
    g_in_dims_collection = {
        "32": [g_conv_dim * 4, g_conv_dim * 4, g_conv_dim * 4],
        "64": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
        "128": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
        "256": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
        "512": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim]
    }
    
    g_out_dims_collection = {
        "32": [g_conv_dim * 4, g_conv_dim * 4, g_conv_dim * 4],
        "64": [g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
        "128": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
        "256": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
        "512": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim, g_conv_dim]
    }
    
    attn_g_location_collection = {
        "32": [2], # apply attention at resolution 16
        "64": [3], # apply attention at resolution 32
        "128": [4], # apply attention at resolution 64
        "256": [4], # apply attention at resolution 64
        "512": [4], # apply attention at resolution 64
    }

    bottom_collection = {"32": 4, "64": 4, "128": 4, "256": 4, "512": 4}
    
    in_dims = g_in_dims_collection[str(img_size)]
    out_dims = g_out_dims_collection[str(img_size)]
    bottom = bottom_collection[str(img_size)]
    attn_g_loc = attn_g_location_collection[str(img_size)]
    num_blocks = len(out_dims)
    
    #### Model ####
    
    # Input z-vector.
    z = tfkl.Input((latent_dim,), name='latent')
    
    # Input class label.
    y = tfkl.Input((1,), dtype=tf.int32, name='class_label')
    
    y_emb = tfkl.Embedding(num_classes, shared_dim)(y)
    y_emb = tf.squeeze(y_emb, axis=1)
    
    if hier:
        c = tfkl.Concatenate()([z, y_emb])
        h = tf.identity(c)
    else:
        c = y_emb
        h = z
    
    # First linear layer + Reshape
    h = SDense(units=in_dims[0]*bottom*bottom, is_sn=True, name='first_linear')(h)
    h = tfkl.Reshape((bottom, bottom, -1), name='first_reshape')(h)
    
    # ResBlocks + Attention
    for index in range(num_blocks):
        for g_index in range(g_depth):
            h = G_Resblock_Deep(h,
                                c,
                                out_channels=in_dims[index] if g_index == 0 else out_dims[index],
                                upsample=True if g_index == (g_depth - 1) else False,
                                studiogan=studiogan,
                                concat=concat,
                                momentum=momentum,
                                epsilon=bn_epsilon,
                                i=f'{index}_{g_index}')
            
        if index + 1 in attn_g_loc and apply_attn:
            h = SelfAttention(name=f'non_local_{index}')(h)            
    
    # output layer: batchnorm-relu-conv.
    h = tfkl.BatchNormalization(epsilon=bn_epsilon, momentum=momentum, name='last_bn')(h) # Not Conditional BatchNorm
    h = tfkl.ReLU(name='last_act')(h)
    h = Conv(filters=out_channels, kernel_size=(3,3), strides=(1, 1), padding=1, is_sn=True, name='last_conv')(h)
    
    # Output activation
    h = tfkl.Activation(output_activation, dtype="float32", name='output_activation')(h)

    # Return Keras model.
    return tfk.Model([z, y], h, name="GeneratorDeep")

def Discriminator(
    num_classes,
    input_shape,
    apply_attn = False,
    wide = True,
    blur_resample = False
):
    # Input image   
    image = tfkl.Input(input_shape, name='image')
    
    # Input class label.
    y = tfkl.Input((1,), dtype=tf.int32, name='class_label')
    
    # ResBlocks + Attention
    h = D_Resblock(image, out_channels=256, is_first=True, is_down=True, blur_resample=blur_resample, wide=wide, i=1)
    if apply_attn:
        h = SelfAttention(name='non_local')(h)
    h = D_Resblock(h, out_channels=256, is_down=True, blur_resample=blur_resample, wide=wide, i=2)
    h = D_Resblock(h, out_channels=256, is_down=False, blur_resample=blur_resample, wide=wide, i=3)
    h = D_Resblock(h, out_channels=256, is_down=False, blur_resample=blur_resample, wide=wide, i=4)
    
    # Apply global sum pooling as in SN-GAN
    h = tfkl.ReLU(name='last_act')(h)
    h = GlobalSumPooling2D(name='global_sum_pooling')(h)
    
    # Get initial class-unconditional output
    out = SDense(units=1, is_sn=True)(h)
    
    y_emb = SEmbedding(num_classes=num_classes, embedding_size=h.shape[-1], is_sn=True)(y)

    # Get projection of final featureset onto class vectors and add to evidence
    #out = out + tf.math.reduce_sum(y_emb * h, axis=1, keepdims=True)
    out = tfkl.Add(name='last_add', dtype='float32')([out, tfkl.Dot((1, 1), name='last_mul')([y_emb, h])]) # same as the commented line above

    # Return Keras model.
    return tfk.Model([image, y], out, name="Discriminator")

def DiscriminatorDeep(
    num_classes,
    input_shape,
    d_conv_dim = 128,
    d_depth = 2,
    apply_attn = True,
    studiogan = True,
    concat = False,
    blur_resample = False
):
    
    #### Configs ####
    d_in_dims_collection = {
        "32": [d_conv_dim if studiogan else d_conv_dim * 4, d_conv_dim * 4, d_conv_dim * 4],
        "64": [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8],
        "128": [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 16],
        "256": [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 8, d_conv_dim * 16],
        "512": [d_conv_dim, d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 8, d_conv_dim * 16]
    }

    d_out_dims_collection = {
        "32": [d_conv_dim * 4, d_conv_dim * 4, d_conv_dim * 4],
        "64": [d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 16],
        "128": [d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 16, d_conv_dim * 16],
        "256": [d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 8, d_conv_dim * 16, d_conv_dim * 16],
        "512": [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 8, d_conv_dim * 16, d_conv_dim * 16]
    }
    
    attn_d_location_collection = {
        "32": [1], # apply attention at resolution 16
        "64": [1], # apply attention at resolution 32
        "128": [1], # apply attention at resolution 64
        "256": [2], # apply attention at resolution 64
        "512": [3], # apply attention at resolution 64
    }
    
    d_down_collection = {
        "32": [True, True, False],
        "64": [True, True, True, True],
        "128": [True, True, True, True, True],
        "256": [True, True, True, True, True, True],
        "512": [True, True, True, True, True, True, True]
    }

    bottom_collection = {"32": 4, "64": 4, "128": 4, "256": 4, "512": 4}
    
    img_size = input_shape[0]
    
    in_dims = d_in_dims_collection[str(img_size)]
    out_dims = d_out_dims_collection[str(img_size)]
    attn_d_loc = attn_d_location_collection[str(img_size)]
    d_down = d_down_collection[str(img_size)]
    num_blocks = len(out_dims)
    
    #### Model ####
    
    # Input image   
    image = tfkl.Input(input_shape, name='image')
    
    # Input class label.
    y = tfkl.Input((1,), dtype=tf.int32, name='class_label')
       
    # Input conv
    h = Conv(filters=in_dims[0], kernel_size=(3,3), strides=(1, 1), padding=1, is_sn=True, name='first_conv')(image)
    
    # ResBlocks + Attention          
    for index in range(num_blocks):
        for d_index in range(d_depth):
            h = D_Resblock_Deep(h,
                                out_channels=out_dims[index],
                                downsample=True if d_down[index] and d_index == 0 else False,
                                is_first=index == 0 and d_index == 0,
                                studiogan=studiogan,
                                concat=concat,
                                blur_resample=blur_resample,
                                i=f'{index}_{d_index}')
            
        if index + 1 in attn_d_loc and apply_attn:
            h = SelfAttention(name=f'non_local_{index}')(h)
    
    # Apply global sum pooling as in SN-GAN
    h = tfkl.ReLU(name='last_act')(h)
    h = GlobalSumPooling2D(name='global_sum_pooling')(h)
    
    # Get initial class-unconditional output
    adv_output = SDense(units=1, is_sn=True)(h)
    
    # Get label embedding
    y_emb = SEmbedding(num_classes=num_classes, embedding_size=h.shape[-1], is_sn=True)(y)

    # Get projection of final featureset onto class vectors and add to evidence
    #out = adv_output + tf.math.reduce_sum(y_emb * h, axis=1, keepdims=True)
    out = tfkl.Add(name='last_add', dtype='float32')([adv_output, tfkl.Dot((1, 1), name='last_mul')([y_emb, h])]) # same as the commented line above

    # Return Keras model.
    return tfk.Model([image, y], out, name="DiscriminatorDeep")

############# Utils #############

def get_generator(cfgs, dataset):
    if cfgs.MODEL.name == 'biggan':
        generator = Generator(
            latent_dim = cfgs.MODEL.latent_dim,
            num_classes = dataset.num_classes,
            hier = cfgs.MODEL.hier,
            apply_attn = cfgs.MODEL.apply_g_attn,
            bn_epsilon = 1e-5,
            momentum = 0.9,
            out_channels = dataset.num_channels,
            output_activation = 'tanh' if dataset.normalize else 'sigmoid'
        )
    elif cfgs.MODEL.name == 'biggan_deep':
        generator = GeneratorDeep(
            latent_dim = cfgs.MODEL.latent_dim,
            shared_dim = cfgs.MODEL.shared_dim,
            num_classes = dataset.num_classes,
            img_size = dataset.image_size,
            hier = cfgs.MODEL.hier,
            g_conv_dim = cfgs.MODEL.g_conv_dim,
            g_depth = cfgs.MODEL.g_depth,
            apply_attn = cfgs.MODEL.apply_g_attn,
            studiogan = cfgs.MODEL.studiogan,
            concat = cfgs.MODEL.residual_concat,
            bn_epsilon = 1e-5,
            momentum = 0.9,
            out_channels = dataset.num_channels,
            output_activation = 'tanh' if dataset.normalize else 'sigmoid'
        )
    else:
        raise NotImplementedError(f"Generator for {cfgs.MODEL.name} not implemented")
    
    return generator

def get_discriminator(cfgs, dataset):
    if cfgs.MODEL.name == 'biggan':
        discriminator = Discriminator(
            num_classes = dataset.num_classes,
            input_shape = (dataset.image_size, dataset.image_size, dataset.num_channels),
            apply_attn = cfgs.MODEL.apply_d_attn,
            wide = cfgs.MODEL.d_wide,
            blur_resample = cfgs.MODEL.blur_resample
        )
    elif cfgs.MODEL.name == 'biggan_deep':
        discriminator = DiscriminatorDeep(
            num_classes = dataset.num_classes,
            input_shape = (dataset.image_size, dataset.image_size, dataset.num_channels),
            d_conv_dim = cfgs.MODEL.d_conv_dim,
            d_depth = cfgs.MODEL.d_depth,
            apply_attn = cfgs.MODEL.apply_d_attn,
            studiogan = cfgs.MODEL.studiogan,
            concat = cfgs.MODEL.residual_concat,
            blur_resample = cfgs.MODEL.blur_resample
        )
    else:
        raise NotImplementedError(f"Discriminator for {cfgs.MODEL.name} not implemented")
    
    return discriminator
