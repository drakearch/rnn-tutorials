from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.rnn_cell_impl import RNNCell
from tensorflow.python.ops.rnn_cell_impl import _linear
from tensorflow.python.ops import variable_scope as vs

num_epochs = 2000
truncated_backprop_length = 15
state_size = 8 #4
num_classes = 2
echo_step = 3
batch_size = 8

class BasicRNNCell(RNNCell):
  """The most basic RNN cell.

  Args:
    num_units: int, The number of units in the LSTM cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
  """

  def __init__(self, num_units, activation=None, reuse=None):
    super(BasicRNNCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or math_ops.tanh

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
    output = self._activation(_linear([inputs, state], self._num_units, True))
    return output, output

class AttentionCell(RNNCell):
    def __init__(self, cell, memory, activation=None, reuse=None):
        super(AttentionCell, self).__init__(_reuse=reuse)
        self._cell = cell
        self._memory = memory
        self._activation = activation or math_ops.tanh
        print(self._memory)

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def call(self, inputs, state):
        N = self._memory.get_shape().as_list()[0]
        L = self._memory.get_shape().as_list()[1]
        k = self._memory.get_shape().as_list()[2]
        eL = tf.ones([L, 1])
        print(state)
        
        inputs1 = tf.matmul(eL, tf.reshape(inputs, shape=[1, -1]))
        inputs1 = tf.transpose(tf.reshape(inputs1, shape=[L, -1, k]), perm=[1, 0, 2])
        inputs1 = tf.reshape(inputs1, shape=[-1, k])
        memory = tf.reshape(self._memory, shape=[-1, k])

        #inputs1 = tf.expand_dims(inputs, 1)
        #inputs1 = tf.concat([tf.matmul(eL, inputs1[i]) for i in range(N)], 0)
        #memory = tf.concat([self._memory[i] for i in range(N)], 0)
 
        inputs1 = self._activation(_linear([memory, inputs1], k, False))
        with vs.variable_scope("Softmax_Attention"):
            inputs1 = _linear([inputs1], 1, False)
            inputs1 = tf.nn.softmax(tf.reshape(inputs1, shape=[-1, L]))

            #inputs1 = tf.matmul(tf.ones([]), tf.reshape(inputs1, shape=[1, -1]))

            inputs1 = tf.expand_dims(inputs1, 1)
        inputs1 = [tf.matmul(inputs1[i], self._memory[i]) for i in range(N)]
        inputs1 = tf.reshape(tf.stack(inputs1, axis=0), shape=[-1, k])
        return self._cell(inputs1, state)

def generateData():
    x = np.array(np.random.choice(2, batch_size * truncated_backprop_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))
    y[:,0:echo_step] = 0
    return (x, y)

batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length,1])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])
seqlen_placeholder = tf.placeholder(tf.int32, [batch_size])

W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

# Unpack columns
inputs_series = tf.unstack(batchX_placeholder, axis=1)
labels_series = tf.unstack(batchY_placeholder, axis=1)
#print("inputs_series: ", inputs_series)
#print()
#print("labels_series: ", labels_series)
#print()

# Forward passes
cell = tf.contrib.rnn.GRUCell(state_size)
cell = AttentionCell(cell, batchX_placeholder)
states_series, current_state = tf.nn.static_rnn(cell, inputs_series, sequence_length=seqlen_placeholder, dtype=tf.float32)
#print("states_series: ", states_series)
#print()
#print("current_series: ", current_state)
#print()

logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
#print("logits_series: ", logits_series)
#print()
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(min(5,batch_size)):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(num_epochs):
        batchX, batchY = generateData()
        batchX = batchX.reshape((batch_size, truncated_backprop_length, -1))
        seqlen = [truncated_backprop_length for i in range(batch_size)]

        _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder: batchX,
                    batchY_placeholder: batchY,
                    seqlen_placeholder: seqlen
                })

        loss_list.append(_total_loss)

        if epoch_idx%100== 0:
            print("Step",epoch_idx, "Loss", _total_loss)
            plot(loss_list, _predictions_series, batchX, batchY)

    print("Step",epoch_idx, "Loss", _total_loss)
    plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()