
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

WEIGHT_INIT_STDDEV = 0.1


class Encoder(object):
    def __init__(self, model_pre_path):
        self.weight_vars = []
        self.model_pre_path = model_pre_path

        with tf.variable_scope('encoder'):
            self.weight_vars.append(self._create_variables(1, 16, 3, scope='conv1_1'))
            self.weight_vars.append(self._create_variables(16, 32, 3, scope='conv1_2'))

    def _create_variables(self, input_filters, output_filters, kernel_size, scope):

        if self.model_pre_path:
            reader = pywrap_tensorflow.NewCheckpointReader(self.model_pre_path)
            with tf.variable_scope(scope):
                kernel = tf.Variable(reader.get_tensor('encoder/' + scope + '/kernel'), name='kernel')
                bias = tf.Variable(reader.get_tensor('encoder/' + scope + '/bias'), name='bias')
        else:
            with tf.variable_scope(scope):
                shape = [kernel_size, kernel_size, input_filters, output_filters]
                kernel = tf.Variable(tf.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV), name='kernel')
                bias = tf.Variable(tf.zeros([output_filters]), name='bias')
        return (kernel, bias)

    def CCM(self, feature, name, ratio=8):

        kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        bias_initializer = tf.constant_initializer(value=0.0)

        with tf.variable_scope(name):
            channel = feature.get_shape()[-1]
            avg_pool = tf.reduce_mean(feature, axis=[1, 2], keepdims=True)

            assert avg_pool.get_shape()[1:] == (1, 1, channel)
            avg_pool = tf.layers.dense(inputs=avg_pool,
                                       units=channel // ratio,
                                       activation=tf.nn.relu,
                                       kernel_initializer=kernel_initializer,
                                       bias_initializer=bias_initializer,
                                       name='mlp_0',
                                       reuse=None)
            assert avg_pool.get_shape()[1:] == (1, 1, channel // ratio)
            avg_pool = tf.layers.dense(inputs=avg_pool,
                                       units=channel,
                                       kernel_initializer=kernel_initializer,
                                       bias_initializer=bias_initializer,
                                       name='mlp_1',
                                       reuse=None)
            assert avg_pool.get_shape()[1:] == (1, 1, channel)

            max_pool = tf.reduce_max(feature, axis=[1, 2], keepdims=True)
            assert max_pool.get_shape()[1:] == (1, 1, channel)
            max_pool = tf.layers.dense(inputs=max_pool,
                                       units=channel // ratio,
                                       activation=tf.nn.relu,
                                       name='mlp_0',
                                       reuse=True)
            assert max_pool.get_shape()[1:] == (1, 1, channel // ratio)
            max_pool = tf.layers.dense(inputs=max_pool,
                                       units=channel,
                                       name='mlp_1',
                                       reuse=True)

            assert max_pool.get_shape()[1:] == (1, 1, channel)
            scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')

        return feature * scale
    def SPCM(self, feature, name):
        kernel_size = 7
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        with tf.variable_scope(name):
            avg_pool = tf.reduce_mean(feature,axis=[3],keepdims=True)
            assert avg_pool.get_shape()[-1] == 1
            max_pool = tf.reduce_max(feature, axis=[3],keepdims=True)
            assert max_pool.get_shape()[-1] == 1
            concat = tf.concat([avg_pool,max_pool],3)
            assert concat.get_shape()[-1] == 2

            concat = tf.layers.conv2d(concat,
                                      filters=1,
                                      kernel_size=[kernel_size,kernel_size],
                                      strides=[1,1],
                                      padding = "same",
                                      activation=None,
                                      kernel_initializer=kernel_initializer,
                                      use_bias = False,
                                      name='conv')
            assert concat.get_shape()[-1] == 1
            concat = tf.sigmoid(concat, 'sigmoid')
        return feature * concat


    def CLCM(self, image, name, ratio=8):

        with tf.variable_scope(name):
            attention_feature = self.CCM(self, image, 'ch_at', ratio)
            attention_feature = self.SPCM(attention_feature, 'sp_at')
            return attention_feature






def conv2d(x, kernel, bias, use_relu=True):
    # padding image with reflection mode
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

    # conv and add bias
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias)

    if use_relu:
        out = tf.nn.relu(out)

    return out

