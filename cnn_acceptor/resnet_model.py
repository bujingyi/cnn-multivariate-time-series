from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

def batch_norm_relu(inputs, is_training, data_format):
    """Performs a batch normalization followed by a ReLU."""
    inputs = tf.layers.batch_normalization(
            inputs=inputs, axis=1 if data_format=='channels_first' else 2,
            momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
            scale=True, training=is_training)
    inputs = tf.nn.relu(inputs)
    return inputs


def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size."""
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
  
    if data_format == 'channels_first':
        paded_inputs = tf.pad(inputs, [[0,0], [0,0], [pad_beg, pad_end]])
    else:
        paded_inputs = tf.pad(inputs, [[0,0], [pad_beg, pad_beg], [0,0]])
    
    return paded_inputs


def conv1d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    """Strided 1-D convolution with explicit padding."""
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)
        
    return tf.layers.conv1d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, 
                            padding=('SAME' if strides == 1 else 'VALID'),
                            use_bias=False,
                            kernel_initializer=tf.variance_scaling_initializer(),
                            data_format=data_format)


def building_block(inputs, filters, is_training, projection_shortcut, strides, data_format):
    """Standard building block for residual networks with BN before convolutions."""
    shortcut = inputs
    inputs = batch_norm_relu(inputs, is_training, data_format)
    
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
    
    inputs = conv1d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, 
                                 strides=strides, data_format=data_format)
    
    inputs = batch_norm_relu(inputs=inputs, is_training=is_training, data_format=data_format)    
    inputs = conv1d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, 
                                  stride=strides, data_format=data_format)
    
    return inputs + shortcut


def bottleneck_block(inputs, filters, is_training, projection_shortcut, strides, data_format):
    """Bottleneck block variant for residual networks with BN before convolutions."""
    shortcut = inputs
    inputs = batch_norm_relu(inputs, is_training, data_format)
    
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
    
    inputs = conv1d_fixed_padding(inputs=inputs, filters=filters, kernel_size=1,
                                  strides=1, data_format=data_format)
    
    inputs = batch_norm_relu(inputs, is_training, data_format)    
    inputs = conv1d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3,
                                  strides=strides, data_format=data_format)
    
    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = conv1d_fixed_padding(inputs=inputs, filters=4 * filters, kernel_size=1,
                                  strides=1, data_format=data_format)
    
    return inputs + shortcut


def block_layer(inputs, filters, block_fn, blocks, strides, is_training, name, data_format):
    """Creates one layer of blocks for the ResNet model."""
    filters_out = 4 * filters if block_fn is bottleneck_block else filters
    
    def projection_shortcut(inputs):
        return conv1d_fixed_padding(inputs=inputs, filters=filters_out, kernel_size=1,
                                    strides=strides, data_format=data_format)
    
    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, is_training, projection_shortcut, strides, data_format)
    
    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, is_training, None, 1, data_format)
    
    return tf.identity(inputs, name)


def n4b_resnet_v2_generator(block_fn, layers, out_width, data_format=None):
    """Generator for ImageNet ResNet v2 models."""
    if data_format is None:
        data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
    
    # print(data_format)
    
    def model(inputs, is_training):
        """Constructs the ResNet model given the inputs."""
        # initial in_width is 128
        # in_width 128 --> 64
        # print(inputs.shape)
        inputs = conv1d_fixed_padding(inputs=inputs, filters=64, kernel_size=7, 
                                      strides=2, data_format=data_format)
        inputs = tf.identity(inputs, 'initial_conv')
        # print('initial_conv:', inputs.shape)
        # in_width 64 --> 32
        inputs = tf.layers.max_pooling1d(inputs=inputs, pool_size=3, strides=2,
                                         padding='SAME', data_format=data_format)
        inputs = tf.identity(inputs, 'initial_max_pool')
        # print('initial_max_pool:', inputs.shape)
        
        # in_width 32 --> 32
        inputs = block_layer(inputs=inputs, filters=64, block_fn=block_fn, 
                             blocks=layers[0], strides=1, is_training=is_training,
                             name='block_layer1', data_format=data_format)
        # print('block_layer1:', inputs.shape)
        # in_width 32 --> 16
        inputs = block_layer(inputs=inputs, filters=128, block_fn=block_fn,
                             blocks=layers[1], strides=2, is_training=is_training,
                             name='block_layer2', data_format=data_format)
        # print('block_layer2:', inputs.shape)
        # in_width 16 --> 8
        inputs = block_layer(inputs=inputs, filters=256, block_fn=block_fn,
                             blocks=layers[2], strides=2, is_training=is_training,
                             name='block_layer3', data_format=data_format)
        # print('block_layer3:', inputs.shape)
        # # in_width 8 --> 4
        # inputs = block_layer(inputs=inputs, filters=512, block_fn=block_fn,
        #                     blocks=layers[3], strides=2, is_training=is_training,
        #                     name='block_layer4', data_format=data_format)
        # print('block_layer4:', inputs.shape)

        # inputs = batch_norm_relu(inputs, is_training, data_format)
        # inputs = tf.layers.average_pooling1d(inputs=inputs, pool_size=2, strides=1,
        #                                     padding='VALID', data_format=data_format)
        # inputs = tf.identity(inputs, 'final_avg_pool')
        # print('final_avg_pool:', inputs.shape)
        inputs = tf.reshape(inputs, [-1, 4 * 1024])
        inputs = tf.layers.dense(inputs=inputs, units=out_width)
        inputs = tf.identity(inputs, 'final_dense')
        return inputs
    
    return model

def n4b_resnet_v2(resnet_size=50, out_width=1, data_format=None):
    model_params = {50: {'block': bottleneck_block, 'layers': [3, 4, 6, 3]}}
    
    params = model_params[resnet_size]
    return n4b_resnet_v2_generator(params['block'], params['layers'], out_width, data_format)    
    

    