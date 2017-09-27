from __future__ import print_function

import tensorflow as tf
import numpy as np

from keras.preprocessing import sequence
from keras.datasets import imdb

max_features = 20000
maxlen = 400  # cut texts after this number of words (among top max_features most common words)
embedding_dim = 50
keep_prob = 0.5

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding='post', truncating='post')
x_test = sequence.pad_sequences(x_test, maxlen=maxlen, padding='post', truncating='post')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

y_train = y_train.reshape([-1, 1]).astype(np.float32)
y_test = y_test.reshape([-1, 1]).astype(np.float32)

class RNN_Model:
    def __init__(self, mode='train'):
        self.x = tf.placeholder(tf.int32, shape=[None, maxlen])
        self.y = tf.placeholder(tf.float32, shape=[None, 1])

        sequence_length = tf.reduce_sum(tf.to_int32(tf.not_equal(self.x, 0)), 1)

        embedding = tf.get_variable('embedding', [max_features, embedding_dim], tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, self.x)

        if mode == 'train':
            inputs = tf.nn.dropout(inputs, keep_prob)

        batch_size = tf.shape(self.x)[0]

        cell = tf.contrib.rnn.BasicLSTMCell(num_units=64, forget_bias=0.0)
        if mode == 'train':
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

        # state = cell.zero_state(batch_size, tf.float32)
        # 
        # with tf.variable_scope("RNN"):
        #     for time_step in range(maxlen):
        #         if time_step > 0:
        #             tf.get_variable_scope().reuse_variables()
        #         (cell_output, state) = cell(inputs[:, time_step, :], state)

        #use dynamic RNN
        initial_state = cell.zero_state(batch_size, tf.float32)
        outputs, state = tf.nn.dynamic_rnn(cell=cell,
                                       inputs=inputs,
                                       sequence_length=sequence_length,
                                       initial_state=initial_state,
                                       dtype=tf.float32)
        cell_output = state[:][1]

        # hidden = tf.layers.dense(cell_output, units=250, activation=tf.nn.relu)
        # if mode == 'train':
        #     hidden = tf.nn.dropout(hidden, keep_prob)

        output_logit = tf.layers.dense(cell_output, units=1)
        self.output = tf.nn.sigmoid(output_logit)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=output_logit))
        self.train = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)

        correct_prediction = tf.equal(self.y, tf.cast(self.output > 0.5, tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.saver = tf.train.Saver()

with tf.variable_scope("model", reuse=None):
    train_model = RNN_Model(mode='train')
with tf.variable_scope("model", reuse = True):
    test_model = RNN_Model(mode='test')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 32
num_epochs = 200
for k in range(num_epochs):
    num_samples = len(y_train)
    random_idx = np.arange(num_samples)
    np.random.shuffle(random_idx)
    x_train_shuffle, y_train_shuffle = x_train[random_idx], y_train[random_idx]
    num_batches = num_samples / batch_size
    for i in range(num_batches):
        x_batch = x_train_shuffle[i * batch_size : (i + 1) * batch_size]
        y_batch = y_train_shuffle[i * batch_size : (i + 1) * batch_size]

        sess.run(train_model.train, feed_dict={train_model.x: x_batch, train_model.y: y_batch})

        num_steps = k * num_batches + i
        if num_steps % 100 == 0:
            train_loss, train_accuracy = sess.run([test_model.loss, test_model.accuracy], feed_dict={test_model.x: x_train, test_model.y: y_train})
            test_loss, test_accuracy = sess.run([test_model.loss, test_model.accuracy], feed_dict={test_model.x: x_test, test_model.y: y_test})

            print('train loss:', train_loss, 'train accuracy:', train_accuracy)
            print('test loss:', test_loss, 'test accuracy:', test_accuracy)

            train_model.saver.save(sess, 'save/imbd_rnn', global_step=train_model.global_step)

sess.close()





