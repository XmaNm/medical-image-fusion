
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
# import tensorflow.keras as keras
import numpy as np
import os
import random

import inference as inference
from tensorflow.examples.tutorials.mnist import input_data
import h5py

import sys, getopt


def run_train(stacks, dataset, sae, ae, noise):

    if dataset == 'mnist':
        # mnist = tf.keras.datasets.mnist
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        shape = int(mnist.validation.images.shape[0] / 2 )
        batch_x1_test = mnist.validation.images[0:shape].reshape(-1, 28, 28, 1)
        batch_x2_test = mnist.validation.images[shape:2*shape].reshape(-1, 28, 28, 1)

        train_images = mnist.train.images.reshape(-1, 28, 28, 1)
        #test_images = mnist.test.images.reshape(-1, 28, 28, 1)
        p_size = train_images.shape[1]

        max_size_h5 = mnist.train.images.shape[0]

    else:
        #filename = 'E:\\Datasets\\32_32_VOC_zero_mean_4mul_imsift_final.h5'
        filename = dataset

        f_h5 = h5py.File(filename, 'r')  # , driver='core'

        train_images = f_h5['train_dataset']
        test_images = f_h5['test_dataset']

        batch_x1_test = test_images[0:500]
        batch_x2_test = test_images[500:1000]

        p_size = train_images.shape[1]
        max_size_h5 = train_images.shape[0]


    batch_size = 32
    max_stack = stacks # Autoencoder stacking
    steps = 15001
    finetune_steps = 20001
    learning_rate = 0.001
    compute_mse = ae # BASIC MSE AUTOENCODER
    compute_mse_diff = sae # SIAMESE AUTOENCODER WITH MANIFOLD LEARNING

    with_noise = noise # If to include noisy data in training (next X steps)
    with_noise_finetune = noise # with noise only during finetune phase (override classic)


    to_compute = np.arange(max_size_h5).tolist()
    steps_togo = steps

    if compute_mse:
        for stack in range(-1, max_stack+1):
            stack = stack+1 # indexing from 1

            tf.reset_default_graph()
            sess = tf.InteractiveSession()
            saver = None

            # setup siamese network
            siamcoder = None
            if stack > max_stack :
                with_noise_togo = with_noise_finetune
                steps_togo = finetune_steps
                siamcoder = inference.siamcoder(stack=stack-1, maxstack=max_stack, mode=tf.estimator.ModeKeys.TRAIN, finetune=True, psize=p_size)
                saver = tf.train.Saver(siamcoder.reuse_list)
            else:
                with_noise_togo = with_noise
                siamcoder = inference.siamcoder(stack=stack, maxstack=max_stack, mode=tf.estimator.ModeKeys.TRAIN, psize=p_size)
                if siamcoder.reuse_list_load:
                    saver = tf.train.Saver(siamcoder.reuse_list_load)


            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(siamcoder.loss_mse)
                train_step_mse_diff = tf.train.RMSPropOptimizer(learning_rate).minimize(siamcoder.loss_mse_diff)


            merged = tf.summary.merge_all()
            # just change overall loss
            with tf.name_scope('loss_test/'):
                loss_test = tf.summary.scalar('loss_test/', siamcoder.loss_mse)
                loss_test_mse = tf.summary.scalar('test_mse', siamcoder.loss_mse)
                loss_test_mse_diff = tf.summary.scalar('test_mse_diff', siamcoder.loss_mse_diff)
            # This is after merge cause we need to add it separately
            with tf.name_scope('loss_train/'):
                loss_train = tf.summary.scalar('loss_train/', siamcoder.loss_mse)
                loss_train_mse = tf.summary.scalar('train_mse', siamcoder.loss_mse)
                loss_train_mse_diff = tf.summary.scalar('train_mse_diff', siamcoder.loss_mse_diff)
            with tf.name_scope('lear_rate'):
                l_r = tf.summary.scalar('l_r', learning_rate)

            file_writer = tf.summary.FileWriter('./results/graph_board_mse' + str(stack), graph=sess.graph)
            tf.initialize_all_variables().run()


            # if you just want to load a previously trainmodel?
            load = False
            model_ckpt = './results/model_mse'+str(stack-1)+'.meta'
            if os.path.isfile(model_ckpt):
                input_var = None
                load = True

            # start training
            if load: saver.restore(sess, './results/model_mse'+str(stack-1))

            # set saver for new variables
            if stack > max_stack:
                saver = tf.train.Saver()
            else:
                saver = tf.train.Saver(siamcoder.reuse_list)



            a = int(batch_size * 0.25)
            b = int(batch_size * 0.2)
            c = int(batch_size * 0.175)
            fr = None
            to = None

            ########
            fr = 0
            to = steps_togo
            loss_t_prev = np.inf
            for step in range(fr, to):#0,steps):
                if len(to_compute) < batch_size:
                    to_compute = np.arange(max_size_h5).tolist()

                # classic way of loading data
                if with_noise_togo:
                    to_compute_batch = [to_compute.pop(random.randrange(len(to_compute))) for _ in range(batch_size)]
                    to_compute_batch = train_images[to_compute_batch]
                    part_a_one = to_compute_batch[0:a]
                    noise_train = np.random.choice([0, 1], size=part_a_one.shape, p=[30 / 100, (100 - 30) / 100])
                    part_b_one = np.multiply(part_a_one, noise_train)

                    part_a_two = to_compute_batch[a:a + b]
                    part_b_two = to_compute_batch[a + b:a + b + b]

                    part_a_three = to_compute_batch[a + 2 * b:a + 2 * b + c]
                    part_b_three = to_compute_batch[a + 2 * b + c:a + 2 * b + 2 * c]

                    batch_x1_o = np.concatenate((part_a_one, part_a_two, part_a_three), axis=0)
                    batch_x2_o = np.concatenate((part_a_one, part_b_two, part_b_three), axis=0)

                    noise_train = np.random.choice([0, 1], size=part_b_three.shape, p=[30 / 100, (100 - 30) / 100])
                    part_b_three = np.multiply(part_b_three, noise_train)

                    batch_x1 = np.concatenate((part_a_one, part_a_two, part_a_three), axis=0)
                    batch_x2 = np.concatenate((part_b_one, part_b_two, part_b_three), axis=0)

                else:
                    to_compute_batch = [to_compute.pop(random.randrange(len(to_compute))) for _ in range(batch_size)]
                    batch_x1 = train_images[to_compute_batch[0:int(batch_size / 2)]]
                    batch_x1_o = train_images[to_compute_batch[0:int(batch_size / 2)]]
                    batch_x2 = train_images[to_compute_batch[int(batch_size / 2):batch_size]]
                    batch_x2_o = train_images[to_compute_batch[int(batch_size / 2):batch_size]]

                _ , loss_v, loss_train_sum, loss_train_sum_mse, loss_train_sum_mse_diff = sess.run(
                    [train_step, siamcoder.loss_mse, loss_train, loss_train_mse, loss_train_mse_diff],
                                    feed_dict={
                                    siamcoder.x1: batch_x1,
                                    siamcoder.x2: batch_x2,
                                    siamcoder.x1_o: batch_x1_o,
                                    siamcoder.x2_o: batch_x2_o})

                if np.isnan(loss_v):
                    print('Model diverged with loss = NaN')
                    break
                    #quit()

                if step % 100 == 0:
                    print ('step %d: loss %.3f' % (step, loss_v))

                if step % 1000 == 0 and step > 0:
                    loss_t, loss_test_sum, loss_test_sum_mse, loss_test_sum_mse_diff, summary = sess.run(
                        [siamcoder.loss_mse, loss_test, loss_test_mse, loss_test_mse_diff, merged], feed_dict={
                        siamcoder.x1: batch_x1_test,
                        siamcoder.x2: batch_x2_test,
                        siamcoder.x1_o: batch_x1_test,
                        siamcoder.x2_o: batch_x2_test,
                        siamcoder.keep_prob: 1.0,
                        siamcoder.training: False})
                    print('step %d: TEST_loss %.3f' % (step, loss_t))
                    file_writer.add_summary(summary, step)
                    # Testing losses
                    file_writer.add_summary(loss_test_sum, step)
                    file_writer.add_summary(loss_test_sum_mse, step)
                    file_writer.add_summary(loss_test_sum_mse_diff, step)
                    # Training losses
                    file_writer.add_summary(loss_train_sum, step)
                    file_writer.add_summary(loss_train_sum_mse, step)
                    file_writer.add_summary(loss_train_sum_mse_diff, step)

                    if loss_t > loss_t_prev * 2.0:
                        print('loss bigger than previous TEST loss. stopping now.')
                        break

                    saver.save(sess, './results/model_mse'+str(stack), write_meta_graph=True)

                    loss_t_prev = loss_t

    if compute_mse_diff:
        steps_togo = steps
        for stack in range(-1, max_stack+1):
            stack = stack+1 #indexing from 1

            tf.reset_default_graph()
            sess = tf.InteractiveSession()
            saver = None

            # setup siamese network
            siamcoder = None
            if stack > max_stack :
                with_noise_togo = with_noise_finetune
                steps_togo = finetune_steps
                siamcoder = inference.siamcoder(stack=stack-1, maxstack=max_stack, mode=tf.estimator.ModeKeys.TRAIN, finetune=True, psize=p_size)
                saver = tf.train.Saver(siamcoder.reuse_list)
            else:
                with_noise_togo = with_noise
                siamcoder = inference.siamcoder(stack=stack, maxstack=max_stack, mode=tf.estimator.ModeKeys.TRAIN, psize=p_size)
                if siamcoder.reuse_list_load:
                    saver = tf.train.Saver(siamcoder.reuse_list_load)


            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(siamcoder.loss_mse)
                train_step_mse_diff = tf.train.RMSPropOptimizer(learning_rate).minimize(siamcoder.loss_mse_diff)

            merged = tf.summary.merge_all()
            # just change overall loss
            with tf.name_scope('loss_test/'):
                loss_test = tf.summary.scalar('loss_test/', siamcoder.loss_mse_diff)
                loss_test_mse = tf.summary.scalar('test_mse', siamcoder.loss_mse)
                loss_test_mse_diff = tf.summary.scalar('test_mse_diff', siamcoder.loss_mse_diff)
            # This is after merge cause we need to add it separately
            with tf.name_scope('loss_train/'):
                loss_train = tf.summary.scalar('loss_train/', siamcoder.loss_mse_diff)
                loss_train_mse = tf.summary.scalar('train_mse', siamcoder.loss_mse)
                loss_train_mse_diff = tf.summary.scalar('train_mse_diff', siamcoder.loss_mse_diff)
            with tf.name_scope('lear_rate'):
                l_r = tf.summary.scalar('l_r', learning_rate)

            file_writer = tf.summary.FileWriter('./results/graph_board_mse_diff' + str(stack), graph=sess.graph)
            tf.initialize_all_variables().run()


            # if you just want to load a previously trainmodel?
            load = False
            model_ckpt = './results/model_mse_diff'+str(stack-1)+'.meta'
            if os.path.isfile(model_ckpt):
                input_var = None
                load = True

            # start training
            if load: saver.restore(sess, './results/model_mse_diff'+str(stack-1))

            # set saver for new variables
            if stack > max_stack:
                saver = tf.train.Saver()
            else:
                saver = tf.train.Saver(siamcoder.reuse_list)



            a = int(batch_size * 0.25)
            b = int(batch_size * 0.2)
            c = int(batch_size * 0.175)
            fr = None
            to = None
            loss_t_prev = np.inf

            if fr is None:
                fr = 0
                to = steps_togo
            else:
                fr = to
                to = to + to

            for step in range(fr, to): #2*steps,3*steps):
                if len(to_compute) < batch_size:
                    to_compute = np.arange(max_size_h5).tolist()

                # classic way of loading data
                if with_noise_togo:
                    to_compute_batch = [to_compute.pop(random.randrange(len(to_compute))) for _ in
                                        range(batch_size)]
                    to_compute_batch = train_images[to_compute_batch]
                    part_a_one = to_compute_batch[0:a]
                    noise_train = np.random.choice([0, 1], size=part_a_one.shape, p=[30 / 100, (100 - 30) / 100])
                    part_b_one = np.multiply(part_a_one, noise_train)

                    part_a_two = to_compute_batch[a:a + b]
                    part_b_two = to_compute_batch[a + b:a + b + b]

                    part_a_three = to_compute_batch[a + 2 * b:a + 2 * b + c]
                    part_b_three = to_compute_batch[a + 2 * b + c:a + 2 * b + 2 * c]

                    batch_x1_o = np.concatenate((part_a_one, part_a_two, part_a_three), axis=0)
                    batch_x2_o = np.concatenate((part_a_one, part_b_two, part_b_three), axis=0)

                    noise_train = np.random.choice([0, 1], size=part_b_three.shape, p=[30 / 100, (100 - 30) / 100])
                    part_b_three = np.multiply(part_b_three, noise_train)

                    batch_x1 = np.concatenate((part_a_one, part_a_two, part_a_three), axis=0)
                    batch_x2 = np.concatenate((part_b_one, part_b_two, part_b_three), axis=0)
                else:
                    to_compute_batch = [to_compute.pop(random.randrange(len(to_compute))) for _ in range(batch_size)]
                    batch_x1 = train_images[to_compute_batch[0:int(batch_size / 2)]]
                    batch_x1_o = train_images[to_compute_batch[0:int(batch_size / 2)]]
                    batch_x2 = train_images[to_compute_batch[int(batch_size / 2):batch_size]]
                    batch_x2_o = train_images[to_compute_batch[int(batch_size / 2):batch_size]]

                _, loss_v, loss_train_sum, loss_train_sum_mse, loss_train_sum_mse_diff = sess.run(
                    [train_step_mse_diff, siamcoder.loss_mse_diff, loss_train, loss_train_mse, loss_train_mse_diff], #_mse_diff
                    feed_dict={
                        siamcoder.x1: batch_x1,
                        siamcoder.x2: batch_x2,
                        siamcoder.x1_o: batch_x1_o,
                        siamcoder.x2_o: batch_x2_o})

                if np.isnan(loss_v):
                    print('Model diverged with loss = NaN')
                    break
                    #quit()

                if step % 100 == 0:
                    print('step %d: loss %.3f' % (step, loss_v))

                # UPRAVIT LEN KAZDY 1000 KROK a obrazky stale tie iste z TESTU (aspon 100)
                if step % 1000 == 0 and step > 0:
                    loss_t, loss_test_sum, loss_test_sum_mse, loss_test_sum_mse_diff, summary = sess.run(
                        [siamcoder.loss_mse_diff, loss_test, loss_test_mse, loss_test_mse_diff, merged], feed_dict={
                        siamcoder.x1: batch_x1_test,
                        siamcoder.x2: batch_x2_test,
                        siamcoder.x1_o: batch_x1_test,
                        siamcoder.x2_o: batch_x2_test,
                        siamcoder.keep_prob: 1.0,
                        siamcoder.training: False})
                    print('step %d: TEST_loss %.3f' % (step, loss_t))
                    file_writer.add_summary(summary, step)
                    # Testing losses
                    file_writer.add_summary(loss_test_sum, step)
                    file_writer.add_summary(loss_test_sum_mse, step)
                    file_writer.add_summary(loss_test_sum_mse_diff, step)
                    # Training losses
                    file_writer.add_summary(loss_train_sum, step)
                    file_writer.add_summary(loss_train_sum_mse, step)
                    file_writer.add_summary(loss_train_sum_mse_diff, step)

                    if loss_t > loss_t_prev * 2.0:
                        print('loss bigger than previous TEST loss. stopping now.')
                        break

                    saver.save(sess, './results/model_mse_diff' + str(stack), write_meta_graph=True)

                    loss_t_prev = loss_t



def main(argv):
   stacks = 4
   dataset = 'mnist'
   sae = True
   ae = False
   noise = False
   try:
      opts, args = getopt.getopt(argv,"hs:d:SAn",["stacks=","dataset="])
   except getopt.GetoptError:
      print('run.py -s <stacks> -d <dataset> (H5PY format only NHWC) -S (to train SAE) -A (to train AE) -n (to include noise in batch)')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('run.py -s <stacks> -d <dataset>( H5PY format only NHWC) -S (to train SAE) -A (to train AE) -n (to include noise in batch)')
         sys.exit()
      elif opt in ("-s", "--stacks"):
          stacks = arg
      elif opt in ("-d", "--dataset"):
          dataset = arg
      elif opt == '-S':
          sae = True
      elif opt == '-A':
          ae = True
      elif opt == '-n':
          noise = True
   run_train(stacks, dataset, sae, ae, noise)

if __name__ == "__main__":
   main(sys.argv[1:])
