from model.ops import *

NET_SCOPE = 'unary'


def simple_net(input, training, classes, width, height):
    with tf.variable_scope(NET_SCOPE,reuse=tf.AUTO_REUSE):
        with tf.variable_scope('vars'):
            d1w, d1b = conv_variables('d1', 3, 16 - 3)
            d2w, d2b = conv_variables('d2', 16, 64 - 16)

            l1w1, l1b1 = residual_non_bottleneck_1d_variables('l1_1', 64)
            l1w2, l1b2 = residual_non_bottleneck_1d_variables('l1_2', 64)
            l1w3, l1b3 = residual_non_bottleneck_1d_variables('l1_1', 64)
            l1w4, l1b4 = residual_non_bottleneck_1d_variables('l1_2', 64)
            l1w5, l1b5 = residual_non_bottleneck_1d_variables('l1_2', 64)

            d3w, d3b = conv_variables('d3', 64, 128 - 64)

            l2w1, l2b1 = residual_non_bottleneck_1d_variables('l2_1', 128)
            l2w2, l2b2 = residual_non_bottleneck_1d_variables('l2_2', 128)
            l2w3, l2b3 = residual_non_bottleneck_1d_variables('l2_3', 128)
            l2w4, l2b4 = residual_non_bottleneck_1d_variables('l2_4', 128)
            l2w5, l2b5 = residual_non_bottleneck_1d_variables('l2_5', 128)
            l2w6, l2b6 = residual_non_bottleneck_1d_variables('l2_6', 128)
            l2w7, l2b7 = residual_non_bottleneck_1d_variables('l2_7', 128)
            l2w8, l2b8 = residual_non_bottleneck_1d_variables('l2_8', 128)

            u1w, u1b = deconv_variables('u1', 128, 64)
            l3w1, l3b1 = residual_non_bottleneck_1d_variables('l3_1', 64)
            l3w2, l3b2 = residual_non_bottleneck_1d_variables('l3_2', 64)

            u2w, u2b = deconv_variables('u2', 64, 16)
            l4w1, l4b1 = residual_non_bottleneck_1d_variables('l4_1', 16)
            l4w2, l4b2 = residual_non_bottleneck_1d_variables('l4_2', 16)

            u3w, u3b = deconv_variables('u3', 16, classes)

        with tf.variable_scope('ops'):
            d1 = conv_max_downsample(input, d1w, d1b, 'd1', training, 0)
            d2 = conv_max_downsample(d1, d2w, d2b, 'd2', training, 0)

            l1 = residual_non_bottleneck_1d(d2, l1w1, l1b1, 1, training, 'l1_1', 0.03)
            l1 = residual_non_bottleneck_1d(l1, l1w2, l1b2, 1, training, 'l1_2', 0.03)
            l1 = residual_non_bottleneck_1d(l1, l1w3, l1b3, 1, training, 'l1_3', 0.03)
            l1 = residual_non_bottleneck_1d(l1, l1w4, l1b4, 1, training, 'l1_4', 0.03)
            l1 = residual_non_bottleneck_1d(l1, l1w5, l1b5, 1, training, 'l1_5', 0.03)

            d3 = conv_max_downsample(l1, d3w, d3b, 'd3', training, 0)

            l2 = residual_non_bottleneck_1d(d3, l2w1, l2b1, 2, training, 'l2_1', 0.3)
            l2 = residual_non_bottleneck_1d(l2, l2w2, l2b2, 4, training, 'l2_2', 0.3)
            l2 = residual_non_bottleneck_1d(l2, l2w3, l2b3, 8, training, 'l2_3', 0.3)
            l2 = residual_non_bottleneck_1d(l2, l2w4, l2b4, 16, training, 'l2_4', 0.3)
            l2 = residual_non_bottleneck_1d(l2, l2w5, l2b5, 2, training, 'l2_5', 0.3)
            l2 = residual_non_bottleneck_1d(l2, l2w6, l2b6, 4, training, 'l2_6', 0.3)
            l2 = residual_non_bottleneck_1d(l2, l2w7, l2b7, 8, training, 'l2_7', 0.3)
            l2 = residual_non_bottleneck_1d(l2, l2w8, l2b8, 16, training, 'l2_8', 0.3)

            u1 = deconv_upsample(l2, u1w, u1b, [64, height // 4, width // 4], 'u1', training, 0)

            l3 = residual_non_bottleneck_1d(u1, l3w1, l3b1, 1, training, 'l3_1', 0)
            l3 = residual_non_bottleneck_1d(l3, l3w2, l3b2, 1, training, 'l3_2', 0)

            u2 = deconv_upsample(l3, u2w, u2b, [16, height // 2, width // 2], 'u2', training, 0)

            l4 = residual_non_bottleneck_1d(u2, l4w1, l4b1, 1, training, 'l4_1', 0)
            l4 = residual_non_bottleneck_1d(l4, l4w2, l4b2, 1, training, 'l4_2', 0)

            logits = deconv_upsample(l4, u3w, u3b, [classes, height, width], 'u3', training, 0, None, False)

            return logits


def net(input, training, classes, width, height):
    to_return = simple_net(input, training, classes, width, height)

    varlist = (tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, NET_SCOPE) +
               tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS, NET_SCOPE))

    return to_return, varlist