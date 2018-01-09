from __future__ import print_function

import numpy as np
import tensorflow as tf
import random

class DataSet(object):
    def __init__(self, n_samples=5, binary_dim=8):
        self.data = []
        self.labels = []
        self.largest_number = pow(2, binary_dim)
        for i in range(n_samples):
            a_int = np.random.randint(self.largest_number/2)
            b_int = np.random.randint(self.largest_number/2)
            c_int = a_int + b_int
            self.data.append(np.unpackbits(
                np.array([[a_int, b_int]], dtype=np.uint8).T, axis=1))
            self.labels.append(np.unpackbits(
                np.array([c_int], dtype=np.uint8).T))
        self.batch_id = 0
            #print(self.data)
            #print(self.labels)
            
    def next(self, batch_size):
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels

learning_rate = 0.01
training_epochs = 10000
#batch_size = 128
batch_size = 4
display_step = 100

seq_len = 8
n_hidden = 8
batch_seqlen = []
for i in range(batch_size):
    batch_seqlen.append(seq_len)

trainset = DataSet(n_samples=5000, binary_dim=seq_len)
testset = DataSet(n_samples=1000, binary_dim=seq_len)

#x = tf.placeholder(tf.float32, [None, seq_len, 2])
x = tf.placeholder(tf.float32, [None, 2, seq_len])
y = tf.placeholder(tf.float32, [None, seq_len])
#y = tf.placeholder("float", [None, seq_len])

seqlen = tf.placeholder(tf.int32, [None])

def dynamicRNN(x, seqlen):
    x = tf.unstack(x, seq_len, 2)
    print("x_train: ", x)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)
    print("output: ", outputs)
    #outputs = tf.stack(outputs)
    #outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])
    print("output: ", outputs)
    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    print("batch_size: ", batch_size)
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
    print("output: ", outputs)
    return outputs

pred = dynamicRNN(x, seqlen)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_epochs:
        batch_x, batch_y = trainset.next(batch_size)
        print(batch_x)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       seqlen: batch_seqlen})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,
                                                seqlen: batch_seqlen})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y,
                                             seqlen: batch_seqlen})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy
    test_data = testset.data
    test_label = testset.labels
    test_seqlen = testset.seqlen
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                      seqlen: test_seqlen}))