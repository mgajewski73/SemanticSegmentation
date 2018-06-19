import math

import tensorflow as tf
import tensorflow.contrib as tfc


def conv_variables(name, in_channels, out_channels, kernel_size=3):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel_size, kernel_size, in_channels, out_channels],
                            initializer=tfc.layers.xavier_initializer_conv2d())
        b = tf.get_variable('b', [out_channels])
        return w, b


def deconv_variables(name, in_channels, out_channels, kernel_size=3):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel_size, kernel_size, out_channels, in_channels],
                            initializer=tfc.layers.xavier_initializer_conv2d())
        b = tf.get_variable('b', [out_channels])
        return w, b


def residual_non_bottleneck_1d_variables(name, channels, kernel_size=3):
    weights = []
    biases = []

    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        weights.append(tf.get_variable('conv1_w', [kernel_size, 1, channels, channels],
                                       initializer=tfc.layers.xavier_initializer_conv2d()))
        biases.append(tf.get_variable('conv1_b', [channels]))
        weights.append(tf.get_variable('conv2_w', [1, kernel_size, channels, channels],
                                       initializer=tfc.layers.xavier_initializer_conv2d()))
        biases.append(tf.get_variable('conv2_b', [channels]))
        weights.append(tf.get_variable('conv3_w', [kernel_size, 1, channels, channels],
                                       initializer=tfc.layers.xavier_initializer_conv2d()))
        biases.append(tf.get_variable('conv3_b', [channels]))
        weights.append(tf.get_variable('conv4_w', [1, kernel_size, channels, channels],
                                       initializer=tfc.layers.xavier_initializer_conv2d()))
        biases.append(tf.get_variable('conv4_b', [channels]))
        return weights, biases


def conv_max_downsample(input, weights, biases, name, training=False, dropout=0.3, activation=tf.nn.relu,
                        batch_normalization=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        conv = tf.nn.bias_add(tf.nn.conv2d(input, weights, (1, 1, 2, 2), 'SAME', data_format='NCHW'), biases,
                              data_format='NCHW')

        if batch_normalization:
            conv = tf.contrib.layers.batch_norm(conv, fused=True, data_format='NCHW', is_training=training)

        if activation:
            conv = activation(conv)

        pool = tf.nn.max_pool(input, (1, 1, 2, 2), (1, 1, 2, 2), 'SAME', data_format='NCHW')
        output = tf.concat([conv, pool], 1)

        if dropout != 0:
            output = tf.layers.dropout(output, dropout, training=training)

        return output


def deconv_upsample(input, weights, biases, out_shape, name, training=False, dropout=0.3, activation=tf.nn.relu,
                    batch_normalization=True):
    with tf.variable_scope(name):
        out_shape.insert(0, -1)
        w_shape = tf.shape(weights)
        x_shape = tf.shape(input)
        output_shape = tf.stack([x_shape[0], w_shape[2], x_shape[2] * 2, x_shape[3] * 2], 0)

        output = tf.nn.bias_add(
            tf.nn.conv2d_transpose(input, weights, output_shape, (1, 1, 2, 2), 'SAME', data_format='NCHW'), biases,
            data_format='NCHW')

        output = tf.reshape(output, out_shape)

        if batch_normalization:
            output = tf.contrib.layers.batch_norm(output, fused=True, data_format='NCHW', is_training=training)

        if dropout != 0.0:
            output = tf.layers.dropout(output, dropout, training=training)

        if activation:
            return activation(output)
        return output


def residual_non_bottleneck_1d(input, weights, biases, dilated, training, name, dropout=0.3, activation=tf.nn.relu,
                               batch_normalization=True):
    with tf.variable_scope(name):
        output = tf.nn.bias_add(tf.nn.conv2d(input, weights[0], (1, 1, 1, 1), 'SAME', data_format='NCHW'), biases[0],
                                data_format='NCHW')
        if activation:
            output = activation(output)

        output = tf.nn.bias_add(tf.nn.conv2d(output, weights[1], (1, 1, 1, 1), 'SAME', data_format='NCHW'), biases[1],
                                data_format='NCHW')
        if batch_normalization:
            output = tf.contrib.layers.batch_norm(output, fused=True, data_format='NCHW', is_training=training)
        if activation:
            output = activation(output)

        output = tf.nn.bias_add(
            tf.nn.conv2d(output, weights[2], (1, 1, 1, 1), 'SAME', data_format='NCHW', dilations=[1, 1, dilated, 1]),
            biases[2], data_format='NCHW')
        if activation:
            output = activation(output)

        output = tf.nn.bias_add(
            tf.nn.conv2d(output, weights[3], (1, 1, 1, 1), 'SAME', data_format='NCHW', dilations=[1, 1, 1, dilated]),
            biases[3], data_format='NCHW')
        if batch_normalization:
            output = tf.contrib.layers.batch_norm(output, fused=True, data_format='NCHW', is_training=training)

        if dropout != 0.0:
            output = tf.layers.dropout(output, dropout, training=training)
        output = output + input
        if activation:
            return activation(output)
        return output

