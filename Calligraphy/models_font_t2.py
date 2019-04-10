import tensorflow as tf
import tensorflow.contrib.slim as slim

from functools import partial

conv = partial(slim.conv2d, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.02))
dconv = partial(slim.conv2d_transpose, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))
fc = partial(ops.flatten_fully_connected, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))
relu = tf.nn.relu
lrelu = partial(ops.leak_relu, leak=0.2)
batch_norm = partial(slim.batch_norm, decay=0.9, scale=True, epsilon=1e-5, updates_collections=None)
ln = slim.layer_norm


def generator(img, num=100, dim=64, reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    conv_ln_lrelu = partial(conv, normalizer_fn=ln, activation_fn=lrelu, biases_initializer=None)
    fc_ln_lrelu = partial(fc, normalizer_fn=ln, activation_fn=lrelu, biases_initializer=None)

    with tf.variable_scope('generator', reuse=reuse):
        with tf.variable_scope('num', reuse=reuse):
            y = lrelu(conv(img, 1, 3, 2))
            y = conv_ln_lrelu(y, dim, 3, 2)
            y = conv_ln_lrelu(y, dim, 3, 3)
            y1 = conv_ln_lrelu(y, 1, 1, 1)
            num_x = tf.tanh(fc_bn_relu(y1, 1))
        with tf.variable_scope('cmd', reuse=reuse):
            y = conv_ln_lrelu(y, num, 3, 1)
            y = conv_ln_lrelu(y, num, 1, 1)
            y = tf.tanh(fc_bn_relu(y, num*4))
            cmd = tf.reshape(y, [-1, num, 4])
        with tf.variable_scope('img', reuse=reuse):
            y = fc_bn_relu(cmd, 1024)
            y = fc_bn_relu(y, 5 * 5 * dim * 2)
            y = tf.reshape(y, [-1, 5, 5, dim * 2])
            y = dconv_bn_relu(y, dim * 2, 3, 3)
            y = dconv_bn_relu(y, dim * 2, 3, 2)
            img = tf.tanh(dconv(y, 1, 3, 2))
        return img, num_x, cmd


def discriminator(img, dim=64, reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)
    fc_bn_lrelu = partial(fc, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

    with tf.variable_scope('discriminator', reuse=reuse):
        y = lrelu(conv(img, 1, 5, 2))
        y = conv_bn_lrelu(y, dim, 5, 2)
        y = fc_bn_lrelu(y, 1024)
        logit = fc(y, 1)
        return logit


def discriminator_wgan_gp(img, dim=64, reuse=True, training=True):
    conv_ln_lrelu = partial(conv, normalizer_fn=ln, activation_fn=lrelu, biases_initializer=None)
    fc_ln_lrelu = partial(fc, normalizer_fn=ln, activation_fn=lrelu, biases_initializer=None)

    with tf.variable_scope('discriminator', reuse=reuse):
        y = lrelu(conv(img, 1, 3, 2))
        # y = conv_ln_lrelu(y, dim*4, 1, 1)
        y = conv_ln_lrelu(y, dim, 3, 2)
        # y = conv_ln_lrelu(y, dim, 1, 1)
        y = conv_ln_lrelu(y, dim, 3, 3)
        y = fc_ln_lrelu(y, 1024)
        logit = fc(y, 1)
        return logit
