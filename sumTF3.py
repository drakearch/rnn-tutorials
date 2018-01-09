from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class DataSet(object):
    def __init__(self, binary_dim=8):
        self.binary_dim = binary_dim
        self.largest_number = pow(2, self.binary_dim)
        self.batch_id = 0
    def next(self, batch_size=5):
        x = []
        y = []
        for i in range(batch_size):
            a_int = np.random.randint(self.largest_number/2)
            b_int = np.random.randint(self.largest_number/2)
            c_int = a_int + b_int
            x1 = np.unpackbits(np.array([a_int], dtype=np.uint8))
            x2 = np.unpackbits(np.array([b_int], dtype=np.uint8))
            y1 = np.unpackbits(np.array([c_int], dtype=np.uint8))
            x1 = x1[::-1]
            x2 = x2[::-1]
            y1 = y1[::-1]
            x.append(np.array([x1, x2]).T)
            y.append(y1)
        return x, y


def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(min(5, batch_size)):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        #plt.bar(left_offset, batchX[batch_series_idx], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)

def printStats(epoch_idx, total_loss, batchX, predictions_series):
    index = np.random.randint(batch_size)
    x = (batchX[index]).T
    a = [0,0]
    for i,b in enumerate(x):
        for j,k in enumerate(b):
            a[i] += k*pow(2,j)

    one_hot = np.array(predictions_series)[:, index, :]
    y = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot])
    c = 0
    for i,b in enumerate(y):
        c += b*pow(2,i)

    print("Step:",epoch_idx, "Loss:", total_loss, "Example:", a[0], "+", a[1], "=", c)

num_epochs = 250
truncated_backprop_length = 8
state_size = 8
num_classes = 2
batch_size = 256
learning_rate = 1

trainset = DataSet()

batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length, 2])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])
#init_state = tf.placeholder(tf.float32, [batch_size, state_size])
cell_state = tf.placeholder(tf.float32, [batch_size, state_size])
hidden_state = tf.placeholder(tf.float32, [batch_size, state_size])
init_state = tf.contrib.rnn.LSTMStateTuple(cell_state, hidden_state)

W = tf.Variable(2*np.random.random((state_size, num_classes))-1,dtype=tf.float32)
b = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

# Unpack columns
inputs_series = tf.unstack(batchX_placeholder, axis=1)
labels_series = tf.unstack(batchY_placeholder, axis=1)
print("inputs_series: ", inputs_series)
print()
print("labels_series: ", labels_series)
print()

# Forward pass
cell = tf.contrib.rnn.BasicLSTMCell(state_size, state_is_tuple=True)
states_series, current_state = tf.nn.static_rnn(cell, inputs_series, init_state)
print("states_series: ", states_series)
print()
print("current_series: ", current_state)
print()

logits_series = [tf.matmul(state, W) + b for state in states_series] #Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
print("logits_series: ", logits_series)
print()
print("predictions_series: ", predictions_series)
print()

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []
    _current_cell_state = np.zeros((batch_size, state_size))
    _current_hidden_state = np.zeros((batch_size, state_size))
    #_current_state = np.zeros((batch_size, state_size))

    for epoch_idx in range(num_epochs):
        batchX,batchY = trainset.next(batch_size)
        #_current_state = np.zeros((batch_size, state_size))
        _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder:batchX,
                    batchY_placeholder:batchY,
                    cell_state: _current_cell_state,
                    hidden_state: _current_hidden_state
                })
        _current_cell_state, _current_hidden_state = _current_state

        loss_list.append(_total_loss)
        plot(loss_list, _predictions_series, batchX, batchY)

        if epoch_idx%20 == 0:
            printStats(epoch_idx, _total_loss, batchX, _predictions_series)
            
plt.ioff()
plt.show()