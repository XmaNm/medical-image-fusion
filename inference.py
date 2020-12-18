import tensorflow as tf
import tensorlayer as tl

from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import l2_regularizer, l1_regularizer

import keras
from keras.layers import Input, Dense, Flatten, Convolution2D, MaxPooling2D, Conv2D, Conv2DTranspose, Dropout, MaxPool2D
from keras.layers import Reshape
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.initializers import glorot_normal
from keras.models import Sequential
from keras.layers.core import *
from keras.optimizers import SGD, RMSprop
from keras import backend as K

# flatten = tf.compat.v1.layers.Flatten
# l2_regularizer = tf.compat.v1.layers


class siamcoder:

    # Create model
    def __init__(self, stack, maxstack, mode, finetune=False, learn_diff=False, psize=32):
        self.p_size = psize #32
        self.x1 = tf.placeholder(tf.float32, [None, self.p_size, self.p_size, 1], 'x1_input')
        self.x2 = tf.placeholder(tf.float32, [None, self.p_size, self.p_size, 1], 'x2_input')
        self.x1_o = tf.placeholder(tf.float32, [None, self.p_size, self.p_size, 1], 'x1_original')
        self.x2_o = tf.placeholder(tf.float32, [None, self.p_size, self.p_size, 1], 'x2_original')

        self.keep_prob = tf.placeholder_with_default(0.5, shape=(), name='keep_prob')
        self.training = tf.placeholder_with_default(True, shape=(), name='training')

        self.finetune = finetune
        self.learn_diff = learn_diff

        self.init_filt = self.p_size * self.p_size
        self.stack = stack
        self.maxstack = maxstack
        self.trainable = False

        self.mode = mode
        self.is_first = True # indicator for settings for 1st autoencoder ( we dont want to add summaries twice for same values)


        self.middle_mse = tf.placeholder(tf.float32, shape=(), name='middle_mse')
        self.end_mse = tf.placeholder(tf.float32, shape=(), name='end_mse')
        self.diff = tf.placeholder(tf.float32, shape=(), name='diff')

        self.mse_one = tf.placeholder(tf.float32, shape=(), name='reconstruct_one')
        self.mse_two = tf.placeholder(tf.float32, shape=(), name='reconstruct_two')
        self.cross_entropy_one = tf.placeholder(tf.float32, shape=(), name='cross_entropy_one')
        self.cross_entropy_two = tf.placeholder(tf.float32, shape=(), name='cross_entropy_two')

        self.o1_beg = tf.placeholder(tf.float32, shape=(), name='o1_beg')
        self.o1_beg_o = tf.placeholder(tf.float32, shape=(), name='o1_beg_original')
        self.o1_mid = tf.placeholder(tf.float32, shape=(), name='o1_mid')
        self.o1_diff = tf.placeholder(tf.float32, shape=(), name='o1_diff')
        self.o1_end = tf.placeholder(tf.float32, shape=(), name='o1_end')
        self.o1_end_resh = tf.placeholder(tf.float32, shape=(), name='o1_end_resh')
        self.o2_beg = tf.placeholder(tf.float32, shape=(), name='o2_beg')
        self.o2_beg_o = tf.placeholder(tf.float32, shape=(), name='o2_beg_original')
        self.o2_mid = tf.placeholder(tf.float32, shape=(), name='o2_mid')
        self.o2_diff = tf.placeholder(tf.float32, shape=(), name='o2_diff')
        self.o2_end = tf.placeholder(tf.float32, shape=(), name='o2_end')
        self.o2_end_resh = tf.placeholder(tf.float32, shape=(), name='o2_end_resh')

        self.mid_weights = tf.placeholder(tf.float32, shape=(), name='mid_weights')
        self.reuse_list = []
        self.reuse_list_load = []

        # internal settings
        self.do_dropout = True
        self.norm = False
        self.ker_reg = None
        self.bias_reg = None
        self.act_reg = None
        self.act_reg_mid = None
        self.initializer = tf.contrib.layers.xavier_initializer(uniform=True)
        self.do_summary = True
        self.use_bias = True
        self.epsilon = 0.0 # 10e-9


        self.act = tf.nn.sigmoid
        self.act_mid = None
        self.act_end = tf.nn.sigmoid

        self.act_diff = tf.nn.sigmoid
        self.act_diff_last = None
        self.act_last = tf.nn.sigmoid
        self.act_last_use = True

        self.current_max = tf.placeholder_with_default(1.0, shape=(), name='current_max')
        self.current_max_end = tf.placeholder_with_default(1.0, shape=(), name='current_max_end')
        self.max_mid = tf.placeholder(tf.float32, shape=(), name='max_mid')
        self.max_end = tf.placeholder(tf.float32, shape=(), name='max_end')
        self.min_mid = tf.placeholder(tf.float32, shape=(), name='min_mid')
        self.min_end = tf.placeholder(tf.float32, shape=(), name='min_end')

        with tf.variable_scope("siamese", reuse=tf.AUTO_REUSE) as scope:
            tf.summary.image('input_first', self.x1, 4)

            self.o1_beg = flatten(self.x1)
            self.o1_beg_o = flatten(self.x1_o)
            tf.summary.histogram('input_hist',self.o1_beg)
            self.o1_mid = self.network_middle(self.o1_beg, scope)


            if self.learn_diff:
                self.o1_diff = self.network_diff(self.o1_mid, scope)

            self.o1_end = self.network_end(self.o1_mid, scope)
            self.o1_end_resh = tf.reshape(self.o1_end, [-1, self.p_size, self.p_size, 1])
            tf.summary.image('output_first', self.o1_end_resh, 4)
            #tf.summary.histogram('output_hist', self.o1_end) ### TODO UNCOMMENT
            self.is_first = False
            scope.reuse_variables()

            self.o2_beg = flatten(self.x2)
            self.o2_beg_o = flatten(self.x2_o)
            self.o2_mid = self.network_middle(self.o2_beg, scope)

            if self.learn_diff:
                self.o2_diff = self.network_diff(self.o2_mid, scope)

            self.o2_end = self.network_end(self.o2_mid, scope)
            self.o2_end_resh = tf.reshape(self.o2_end, [-1, self.p_size, self.p_size, 1])


            # Create loss
            self.y_ = tf.placeholder(tf.float32, [None])


            if learn_diff:
                self.loss_diff = self.siamcoder_loss_diff()
            self.loss_mse_diff = self.siamcoder_loss_mse_diff()
            self.loss_mse = self.siamcoder_loss_mse()

            with tf.name_scope('diffs'):
                tf.summary.scalar('diff', self.diff)
                tf.summary.scalar('mse_one', self.mse_one)
                tf.summary.scalar('mse_two', self.mse_two)

            with tf.name_scope('layers_mean'):
                tf.summary.scalar('o1_mid', tf.reduce_mean(self.o1_mid))
                tf.summary.scalar('o1_end', tf.reduce_mean(self.o1_end))
                tf.summary.scalar('o2_mid', tf.reduce_mean(self.o2_mid))
                tf.summary.scalar('o2_end', tf.reduce_mean(self.o2_end))

    def add_to_reuse_list(self, var):
        if self.is_first:
            self.reuse_list.append(var)

    def add_to_reuse_list_load(self, var):
        if self.is_first:
            self.reuse_list_load.append(var)

    def test(self):
        self.a = 1

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))

    def layer_summary(self, layer_name, input_tensor, output):
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = tf.get_variable(layer_name+'/kernel')
                self.variable_summaries(var=weights)
            if self.use_bias:
                with tf.name_scope('biases'):
                    biases = tf.get_variable(layer_name+'/bias')
                    self.variable_summaries(var=biases)

    def return_true(self):
        self.trainable = True
        return True

    def return_false(self):
        self.trainable = False
        return False

    def network_middle(self, x, scope):
        fc = x

        self.trainable = self.finetune
        if 0 == self.stack: #1 == self.stack or
            self.trainable = True

        if self.learn_diff:
            self.trainable = False

        input = fc
        fc = tf.layers.dense(inputs=input, units=self.init_filt, activation=self.act,
                             trainable=self.trainable, name='fc_first',
                             kernel_initializer=self.initializer,
                             kernel_regularizer=self.ker_reg,
                             bias_regularizer=self.bias_reg,
                             activity_regularizer=self.act_reg,
                             use_bias=self.use_bias)

        self.add_to_reuse_list(tf.get_variable('fc_first/kernel'))
        if self.use_bias:
            self.add_to_reuse_list(tf.get_variable('fc_first/bias'))
        if self.trainable == False: # UPDATE MJ 30.4.2018
            self.add_to_reuse_list_load(tf.get_variable('fc_first/kernel'))
            if self.use_bias:
                self.add_to_reuse_list_load(tf.get_variable('fc_first/bias'))

        # summary
        if self.is_first and self.do_summary:
            self.layer_summary(layer_name='fc_first', input_tensor=input, output=fc)
        if self.norm:
            fc = tf.layers.batch_normalization(fc, name='norm_fc_first', trainable=self.trainable,
                                               training=self.training)


        for i in range(self.stack):
            i = i+1
            input = fc
            units = int(self.init_filt/(pow(2, i-1)))
            units = int(self.init_filt / (pow(2, i - 1)))
            if units < 50:
                units = 2
            fc = tf.layers.dense(inputs=input, units=units, activation=self.act,
                                 trainable=self.trainable, name='fc_'+str(i),
                                 kernel_initializer=self.initializer,
                                 kernel_regularizer=self.ker_reg,
                                 bias_regularizer=self.bias_reg,
                                 activity_regularizer=self.act_reg,
                                 use_bias=self.use_bias)

            self.add_to_reuse_list(tf.get_variable('fc_'+str(i)+'/kernel'))
            if self.use_bias:
                self.add_to_reuse_list(tf.get_variable('fc_'+str(i)+'/bias'))
            if self.trainable == False:
                self.add_to_reuse_list_load(tf.get_variable('fc_' + str(i) + '/kernel'))
                if self.use_bias:
                    self.add_to_reuse_list_load(tf.get_variable('fc_' + str(i) + '/bias'))
            # summary
            if self.is_first and self.do_summary:
                self.layer_summary(layer_name='fc_' + str(i), input_tensor=input, output=fc)
            if self.norm:
                fc = tf.layers.batch_normalization(fc, name='norm_fc_'+str(i), trainable=self.trainable, training=self.training)


        self.trainable = True
        if self.learn_diff:
            self.trainable = False
        input = fc

        if self.stack == self.maxstack:
            units = int(self.init_filt / (pow(2, self.stack)))
            if units < 50:
                units = 2
            fc = tf.layers.dense(inputs=input, units=units, activation=self.act_mid,
                                 trainable=self.trainable, name='fc_'+str(self.stack+1),
                                 kernel_initializer=self.initializer,
                                 kernel_regularizer=self.ker_reg,
                                 bias_regularizer=self.bias_reg,
                                 activity_regularizer=self.act_reg_mid,
                                 use_bias=self.use_bias)
        else:
            units = int(self.init_filt / (pow(2, self.stack)))
            if units < 50:
                units = 2
            fc = tf.layers.dense(inputs=input, units=units,
                                 activation=self.act,
                                 trainable=self.trainable, name='fc_' + str(self.stack + 1),
                                 kernel_initializer=self.initializer,
                                 kernel_regularizer=self.ker_reg,
                                 bias_regularizer=self.bias_reg,
                                 activity_regularizer=self.act_reg_mid,
                                 use_bias=self.use_bias)

        self.add_to_reuse_list(tf.get_variable('fc_'+str(self.stack+1) + '/kernel'))
        if self.use_bias:
            self.add_to_reuse_list(tf.get_variable('fc_'+str(self.stack+1) + '/bias'))

        # summary
        if self.is_first and self.do_summary:
            self.layer_summary(layer_name='fc_' + str(self.stack + 1), input_tensor=input, output=fc)

        if self.norm:
            fc = tf.layers.batch_normalization(fc, name='norm_fc_'+str(self.stack+1), trainable=self.trainable, training=self.training)
            # fc = tf.contrib.layers.layer_norm(fc, scope=scope, trainable=self.trainable)

        if self.do_dropout:
            input = fc
            fc = tf.layers.dropout(inputs=input, rate=self.keep_prob) #, training=self.mode == tf.estimator.ModeKeys.TRAIN)

        return fc

    def network_end(self, fc, scope):

        self.trainable = self.finetune
        if 0 == self.stack: #1 == self.stack or
            self.trainable = True
        if self.learn_diff:
            self.trainable = False

        for i in reversed(range(self.stack)):
            i = i+1
            if i == self.stack:
                self.trainable = True
            else:
                self.trainable = False
                if self.finetune:
                    self.trainable = True

            if self.learn_diff:
                self.trainable = False
            input = fc

            #last layer different act
            if i == 1 and self.act_last_use:
                fc = tf.layers.dense(inputs=input, units=int(self.init_filt / (pow(2, i - 1))), activation=self.act_last,
                                     trainable=self.trainable, name='fc_r' + str(i),
                                     kernel_initializer=self.initializer,
                                     kernel_regularizer=self.ker_reg,
                                     bias_regularizer=self.bias_reg,
                                     activity_regularizer=self.act_reg,
                                     use_bias=self.use_bias)

            else:
                fc = tf.layers.dense(inputs=input, units=int(self.init_filt / (pow(2, i - 1))), activation=self.act_end,
                                     trainable=self.trainable, name='fc_r' + str(i),
                                     kernel_initializer=self.initializer,
                                     kernel_regularizer=self.ker_reg,
                                     bias_regularizer=self.bias_reg,
                                     activity_regularizer=self.act_reg,
                                     use_bias=self.use_bias)

            self.add_to_reuse_list(tf.get_variable('fc_r' + str(i) + '/kernel'))
            if self.use_bias:
                self.add_to_reuse_list(tf.get_variable('fc_r' + str(i) + '/bias'))
            if self.trainable == False:
                self.add_to_reuse_list_load(tf.get_variable('fc_r' + str(i) + '/kernel'))
                if self.use_bias:
                    self.add_to_reuse_list_load(tf.get_variable('fc_r' + str(i) + '/bias'))
            # summary
            if self.is_first and self.do_summary:
                self.layer_summary(layer_name='fc_r' + str(i), input_tensor=input, output=fc)
            if self.norm and i != 1: #we dont need normalization after last layer
                fc = tf.layers.batch_normalization(fc, name='norm_fc_r_' + str(i), trainable=self.trainable, training=self.training)


        input = fc
        fc = tf.layers.dense(inputs=input, units=self.init_filt, activation=self.act_last,
                             trainable=self.trainable, name='fc_last',
                             kernel_initializer=self.initializer,
                             kernel_regularizer=self.ker_reg,
                             bias_regularizer=self.bias_reg,
                             activity_regularizer=self.act_reg,
                             use_bias=self.use_bias)

        self.add_to_reuse_list(tf.get_variable('fc_last/kernel'))
        if self.use_bias:
            self.add_to_reuse_list(tf.get_variable('fc_last/bias'))
        if self.trainable == False:
            self.add_to_reuse_list_load(tf.get_variable('fc_last/kernel'))
            if self.use_bias:
                self.add_to_reuse_list_load(tf.get_variable('fc_last/bias'))
        # summary
        if self.is_first and self.do_summary:
            self.layer_summary(layer_name='fc_last', input_tensor=input, output=fc)
        if self.norm:
            fc = tf.layers.batch_normalization(fc, name='norm_fc_last', trainable=self.trainable, training=self.training)

        return fc

    def network_diff(self, fc, scope):
        input = fc

        fc = tf.layers.dense(inputs=input, units=int(self.init_filt / (pow(2, self.stack))),
                             activation=self.act_diff,
                             trainable=True, name='fc_diff_1',
                             kernel_initializer=self.initializer,
                             kernel_regularizer=self.ker_reg,
                             bias_regularizer=self.bias_reg,
                             activity_regularizer=self.act_reg,
                             use_bias=self.use_bias)

        if self.is_first and self.do_summary:
            self.layer_summary(layer_name='fc_diff_1', input_tensor=input, output=fc)


        if self.norm:
            fc = tf.layers.batch_normalization(fc, name='norm_fc_diff1', trainable=True, training=self.training)

        input = fc

        fc = tf.layers.dense(inputs=input, units=int(self.init_filt / (pow(2, self.stack))),
                             activation=self.act_diff_last,
                             trainable=True, name='fc_diff_2',
                             kernel_initializer=self.initializer,
                             kernel_regularizer=self.ker_reg,
                             bias_regularizer=self.bias_reg,
                             activity_regularizer=self.act_reg,
                             use_bias=self.use_bias)

        if self.is_first and self.do_summary:
            self.layer_summary(layer_name='fc_diff_2', input_tensor=input, output=fc)

        if self.norm:
            fc = tf.layers.batch_normalization(fc, name='norm_fc_diff2', trainable=True, training=self.training)

        if self.use_bias:
            self.add_to_reuse_list_load(tf.get_variable('fc_diff_1/bias'))
            self.add_to_reuse_list_load(tf.get_variable('fc_diff_2/bias'))

        self.add_to_reuse_list_load(tf.get_variable('fc_diff_1/kernel'))
        self.add_to_reuse_list_load(tf.get_variable('fc_diff_2/kernel'))

        return fc

    def fc_layer(self, stack_index, bottom, n_weight, name):
        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc



    def siamcoder_loss_diff(self):

        # COSINE
        normalize_a_diff1 = tf.nn.l2_normalize(self.o1_diff, axis=1)
        normalize_b_diff1 = tf.nn.l2_normalize(self.o2_diff, axis=1)
        cos_similarity_diff1 = 1 - tf.reduce_sum(tf.multiply(normalize_a_diff1, normalize_b_diff1), axis=1)

        normalize_a_diff2 = tf.nn.l2_normalize(self.o1_end, axis=1)
        normalize_b_diff2 = tf.nn.l2_normalize(self.o2_end, axis=1)
        cos_similarity_diff2 = 1 - tf.reduce_sum(tf.multiply(normalize_a_diff2, normalize_b_diff2), axis=1)

        # just to log
        self.middle_mse = tf.reduce_mean(cos_similarity_diff1)
        self.end_mse = tf.reduce_mean(cos_similarity_diff2)
        self.diff = tf.reduce_mean(tf.abs(cos_similarity_diff1 - cos_similarity_diff2))

        normalize_a1 = tf.nn.l2_normalize(self.o1_beg, axis=1)
        normalize_b1 = tf.nn.l2_normalize(self.o1_end, axis=1)
        cos_similarity1 = tf.reduce_mean(1 - tf.reduce_sum(tf.multiply(normalize_a1, normalize_b1), axis=1))

        normalize_a2 = tf.nn.l2_normalize(self.o2_beg, axis=1)
        normalize_b2 = tf.nn.l2_normalize(self.o2_end, axis=1)
        cos_similarity2 = tf.reduce_mean(1 - tf.reduce_sum(tf.multiply(normalize_a2, normalize_b2), axis=1))

        self.mse_one = cos_similarity1
        self.mse_two = cos_similarity2


        return self.diff

    def siamcoder_loss_mse_diff(self):

        # COSINE
        normalize_a_diff1 = tf.nn.l2_normalize(self.o1_mid, axis=1)
        normalize_b_diff1 = tf.nn.l2_normalize(self.o2_mid, axis=1)
        cos_similarity_diff1 = 1 - tf.reduce_sum(tf.multiply(normalize_a_diff1, normalize_b_diff1), axis=1)

        normalize_a_diff2 = tf.nn.l2_normalize(self.o1_end, axis=1)
        normalize_b_diff2 = tf.nn.l2_normalize(self.o2_end, axis=1)
        cos_similarity_diff2 = 1 - tf.reduce_sum(tf.multiply(normalize_a_diff2, normalize_b_diff2), axis=1)

        # just to log
        self.middle_mse = tf.reduce_mean(cos_similarity_diff1)
        self.end_mse = tf.reduce_mean(cos_similarity_diff2)
        self.diff = tf.reduce_mean(tf.abs(cos_similarity_diff1 - cos_similarity_diff2))


        normalize_a1 = tf.nn.l2_normalize(self.o1_beg, axis=1)
        normalize_b1 = tf.nn.l2_normalize(self.o1_end, axis=1)
        cos_similarity1 = tf.reduce_mean(1 - tf.reduce_sum(tf.multiply(normalize_a1, normalize_b1), axis=1))

        normalize_a2 = tf.nn.l2_normalize(self.o2_beg, axis=1)
        normalize_b2 = tf.nn.l2_normalize(self.o2_end, axis=1)
        cos_similarity2 = tf.reduce_mean(1 - tf.reduce_sum(tf.multiply(normalize_a2, normalize_b2), axis=1))

        self.mse_one = cos_similarity1
        self.mse_two = cos_similarity2


        return self.diff + self.mse_one + self.mse_two

    def siamcoder_loss_mse(self):

        normalize_a_diff1 = tf.nn.l2_normalize(self.o1_mid, axis=1)
        normalize_b_diff1 = tf.nn.l2_normalize(self.o2_mid, axis=1)
        cos_similarity_diff1 = 1 - tf.reduce_sum(tf.multiply(normalize_a_diff1, normalize_b_diff1), axis=1)

        normalize_a_diff2 = tf.nn.l2_normalize(self.o1_end, axis=1)
        normalize_b_diff2 = tf.nn.l2_normalize(self.o2_end, axis=1)
        cos_similarity_diff2 = 1 - tf.reduce_sum(tf.multiply(normalize_a_diff2, normalize_b_diff2), axis=1)

        # just to log
        self.middle_mse = tf.reduce_mean(cos_similarity_diff1)
        self.end_mse = tf.reduce_mean(cos_similarity_diff2)
        self.diff = tf.reduce_mean(tf.abs(cos_similarity_diff1 - cos_similarity_diff2))

        normalize_a1 = tf.nn.l2_normalize(self.o1_beg, axis=1)
        normalize_b1 = tf.nn.l2_normalize(self.o1_end, axis=1)
        cos_similarity1 = tf.reduce_mean(1 - tf.reduce_sum(tf.multiply(normalize_a1, normalize_b1),axis=1))

        normalize_a2 = tf.nn.l2_normalize(self.o2_beg, axis=1)
        normalize_b2 = tf.nn.l2_normalize(self.o2_end, axis=1)
        cos_similarity2 = tf.reduce_mean(1 - tf.reduce_sum(tf.multiply(normalize_a2, normalize_b2), axis=1))

        self.mse_one = cos_similarity1
        self.mse_two = cos_similarity2

        return self.mse_one + self.mse_two