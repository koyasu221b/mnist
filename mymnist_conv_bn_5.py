# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import division
from __future__ import print_function
from parser import read_data_sets
from tensorflow.python import control_flow_ops
#from tensorflow.examples.tutorials.mnist import input_data
#google_mnist = input_data.read_data_sets("data", one_hot=True)
import tensorflow as tf
import numpy
import time
from destort import get_generate2
import logging
import logging.config
logging.config.fileConfig('logging.conf')
# create logger
logger = logging.getLogger('simpleExample')


mnist = read_data_sets("data2", one_hot=True)
generate, validate = get_generate2(
                     "data2",
                     "generate_cache.pkl",
                     use=[
                          "morphology",
                          "default",
                          "shift",
                          "shear",
                         "rotate"
                     ])

mnist.generate = generate
mnist.validate = validate

batch_size = 100
learning_rate = 1e-4
simple_validation_size = 100
total_validation_size = 10000


'''
Create the model
x, which is a 784-dim vector, is input layer
W, which is a 784x10 matrix, is the weight of the input layer
b, which is a 784-dim vector, is the bias of the input layer
y, which is a one-hot vector (10-dim), is the prediction
'''

def conv_batch_norm(x, n_out, phase_train):
    beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
    gamma_init = tf.constant_initializer(value=1.0, dtype=tf.float32)

    beta = tf.get_variable("beta", [n_out], initializer=beta_init)
    gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init)

    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = control_flow_ops.cond(phase_train,
        mean_var_with_update,
        lambda: (ema_mean, ema_var))

    normed = tf.nn.batch_norm_with_global_normalization(x, mean, var,
        beta, gamma, 1e-3, True)
    return normed

def layer_batch_norm(x, n_out, phase_train):
    beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
    gamma_init = tf.constant_initializer(value=1.0, dtype=tf.float32)

    beta = tf.get_variable("beta", [n_out], initializer=beta_init)
    gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init)

    batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = control_flow_ops.cond(phase_train,
        mean_var_with_update,
        lambda: (ema_mean, ema_var))

    reshaped_x = tf.reshape(x, [-1, 1, 1, n_out])
    normed = tf.nn.batch_norm_with_global_normalization(reshaped_x, mean, var,
        beta, gamma, 1e-3, True)
    return tf.reshape(normed, [-1, n_out])


def layer(input, weight_shape, bias_shape, phase_train, is_output=False):
    weight_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape,
                        initializer=weight_init)
    b = tf.get_variable("b", bias_shape,
                        initializer=bias_init)
    logits = tf.matmul(input, W) + b
    if is_output:
        return layer_batch_norm(logits, weight_shape[1], phase_train)
    else:
        return tf.nn.relu(layer_batch_norm(logits, weight_shape[1], phase_train))

def conv2d(input, weight_shape, bias_shape, phase_train):
    incoming = weight_shape[0] * weight_shape[1] * weight_shape[2]
    weight_init = tf.random_normal_initializer(stddev=(2.0/incoming)**0.5)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    logits = tf.nn.bias_add(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME'), b)
    return tf.nn.relu(conv_batch_norm(logits, weight_shape[3], phase_train))

def max_pool(input, k=2):
    return tf.nn.max_pool(
        input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def inference(x, keep_prob, phase_train):

    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    with tf.variable_scope("conv_1"):
        conv_1 = conv2d(x, [5, 5, 1, 64], [64], phase_train)
        pool_1 = max_pool(conv_1)

    with tf.variable_scope("conv_2"):
        conv_2 = conv2d(pool_1, [5, 5, 64, 64], [64], phase_train)
        pool_2 = max_pool(conv_2)

    with tf.variable_scope("fc"):
        pool_2_flat = tf.reshape(pool_2, [-1, 7 * 7 * 64])
        pool_2_flat_drop = tf.nn.dropout(pool_2_flat, keep_prob)
        fc_1 = layer(pool_2_flat_drop, [7*7*64, 1024], [1024], phase_train)

        # apply dropout
        fc_1_drop = tf.nn.dropout(fc_1, keep_prob)

    with tf.variable_scope("output"):
        output = layer(fc_1_drop, [1024, 10], [10], phase_train, True)

    return output

def loss(output, y):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(output, y)
    loss = tf.reduce_mean(xentropy)
    return loss

def training(cost, learning_rate, global_step):
    logger.info("learning_rate is %r" % learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(cost, global_step=global_step)
    return train_step

def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def training_process(model_path, total_epoch=1):
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32) # dropout probability
    phase_train = tf.placeholder(tf.bool) # training or testing

    output = inference(x, keep_prob, phase_train)
    cost = loss(output, y_)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_step = training(cost, learning_rate, global_step)
    eval_op = evaluate(output, y_)
    sess = tf.Session()
    init_op = tf.initialize_all_variables()

    sess.run(init_op)
    saver = tf.train.Saver()
    for epoch in range(total_epoch):
        current = int(time.time())
        logger.info("epoch %s " % epoch)
        logger.info("total training data %s" % mnist.generate.num_examples)
        total_batch = int(mnist.generate.num_examples/batch_size)
        total_loss = 0
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.generate.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5, phase_train: True})
            result = sess.run([eval_op, cost], feed_dict={x: mnist.validate.images, y_: mnist.validate.labels, keep_prob: 1.0, phase_train:False})
            accuracy = result[0]
            my_loss = result[1]
            total_loss += my_loss
        print("loss: ", total_loss/total_batch)

    save_path = saver.save(sess, model_path)

    sess.close()


def testing_10w_process(model_path, output_path):
    restore_path = model_path

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32) # dropout probability
    phase_train = tf.placeholder(tf.bool) # training or testing

    output = inference(x, keep_prob, phase_train)
    cost = loss(output, y_)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_step = training(cost, learning_rate, global_step)
    eval_op = evaluate(output, y_)

    sess = tf.Session()
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    saver = tf.train.Saver()
    saver.restore(sess, restore_path)
    b = mnist.test.ids
    logger.info(len(b))
    out_seq = []
    for i in range(10):
        start = i * 10000
        end = (i + 1) * 10000
        eval_feed = {x: mnist.test.images[start:end], keep_prob: 1.0, phase_train: False}
        rrr = sess.run(tf.nn.softmax(output), feed_dict=eval_feed)
        DAT = numpy.column_stack((numpy.array(b[start:end]), rrr))

        out_seq.append(DAT)

    outfile = numpy.row_stack(out_seq)
    numpy.savetxt(output_path, outfile, delimiter=",", fmt="%s")

    sess.close()

if __name__ == '__main__':
    model_path = 'model_conv_bn'
    output_path = 'sample.csv'
    total_epoch=300
    training_process(model_path, total_epoch)
    testing_10w_process(model_path, output_path)
