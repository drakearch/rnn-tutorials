from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_epochs = 100
truncated_backprop_length = 15
state_size = 12
num_classes = 2
echo_step = 3
batch_size = 10

class DataSet(object):
    def __init__(self):
        self.batch_id = 0
    def next(self, batch_size=5):
        x = []
        y = []
        for i in range(batch_size):
            x1 = np.array(np.random.choice(2, truncated_backprop_length, p=[0.5, 0.5]))
            x2 = np.array(np.random.choice(2, truncated_backprop_length, p=[0.5, 0.5]))
            y1 = np.roll(x1, echo_step)
            y1[0:echo_step] = 0
            x.append(np.array([x1, x2]).T)
            y.append(y1)
        return x, y

trainset = DataSet()

batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length, 2])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

init_state = tf.placeholder(tf.float32, [batch_size, state_size])

W = tf.Variable(2*np.random.random((state_size+2, state_size))-1, dtype=tf.float32)
b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)

W2 = tf.Variable(2*np.random.random((state_size, num_classes))-1,dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

# Unpack columns
inputs_series = tf.unstack(batchX_placeholder, axis=1)
labels_series = tf.unstack(batchY_placeholder, axis=1)
print("inputs_series: ", inputs_series)
print()
print("labels_series: ", labels_series)
print()

# Forward pass
current_state = init_state
states_series = []
for current_input in inputs_series:
    current_input = tf.reshape(current_input, [batch_size, -1])
    input_and_state_concatenated = tf.concat([current_input, current_state], 1)  # Increasing number of columns
    print("input_states: ", input_and_state_concatenated)
    print()
    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
    states_series.append(next_state)
    current_state = next_state
print("states_series: ", states_series)
print()
print("current_series: ", current_state)
print()

logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
print("logits_series: ", logits_series)
print()

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

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


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(num_epochs):
        batchX,batchY = trainset.next(batch_size)

        print("New data, epoch", epoch_idx)         
        
        _current_state = np.zeros((batch_size, state_size))

        _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder:batchX,
                    batchY_placeholder:batchY,
                    init_state:_current_state
                })

        loss_list.append(_total_loss)

        if epoch_idx%1 == 0:
            print("Step",epoch_idx, "Loss", _total_loss)
            plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()