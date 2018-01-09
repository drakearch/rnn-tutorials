from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

random.seed()

class DataSet(object):
    def __init__(self, binary_dim=8):
        self.binary_dim = binary_dim
        self.largest_number = pow(2, binary_dim)
        self.batch_id = 0
            
    def next(self, batch_size):
        data = []
        labels=[]
        for i in range(batch_size):
            a_int = np.random.randint(self.largest_number/2)
            b_int = np.random.randint(self.largest_number/2)
            c_int = a_int + b_int
            data.append(np.unpackbits(
                np.array([[a_int, b_int]], dtype=np.uint8), axis=0))
            labels.append(np.unpackbits(
                np.array([[c_int]], dtype=np.uint8), axis=0))
        self.batch_id = self.batch_id+1
        return data, labels

learning_rate = 0.1
training_epochs = 2000000
batch_size = 100
display_step = 100

num_epochs = training_epochs//batch_size
num_batches = 5000//batch_size
num_classes = 1

seq_len = 8
state_size = 16

trainset = DataSet(binary_dim=seq_len)

batchX_placeholder = tf.placeholder(tf.float32, [None, seq_len, 2])
batchY_placeholder = tf.placeholder(tf.float32, [None, seq_len, 1])
init_state = tf.placeholder(tf.float32, [None, state_size])

# +2 for dimension of input
W = tf.Variable(2*np.random.random((state_size+2, state_size))-1, dtype=tf.float32)
b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)

W2 = tf.Variable(2*np.random.random((state_size, num_classes))-1,dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

# Unpack columns
inputs_series = tf.unstack(batchX_placeholder, axis=1)
labels_series = tf.unstack(batchY_placeholder, axis=1)
print(inputs_series)
print()
print(labels_series)

# Forward pass
current_state = init_state
states_series = []
for current_input in inputs_series:
    #current_input = tf.reshape(current_input, [batch_size, 2])
    input_and_state_concatenated = tf.concat([current_input, current_state], 1)  # Increasing number of columns
    print("concat: ", input_and_state_concatenated)
    next_state = tf.sigmoid(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
    states_series.append(next_state)
    current_state = next_state
print(states_series)

logits_series = [tf.sigmoid(tf.matmul(state, W2) + b2) for state in states_series] #Broadcasted addition
#predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
print("logits: ", logits_series)
#labels_series = [tf.reshape(logit, [None, 1]) for logit in labels_series]
losses = [tf.losses.mean_squared_error(labels_series, logits_series)]
#losses = [np.abs(logits-labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_sum(losses)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(min(batch_size,5)):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(0 if out < 0.5 else 1) for out in one_hot_output_series])


        #print("Pred: ",batchY)
        #print()
        #print("logits: ",single_output_series)
        #print()

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, seq_len, 0, 2])
        left_offset = range(seq_len)
        plt.bar(left_offset, batchY[batch_series_idx], width=1, color="blue")
        #plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.7, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(num_epochs):
        _current_state = np.zeros((batch_size, state_size))

        batchX, batchY = trainset.next(batch_size)

        _total_loss, _train_step, _current_state, _logits_series, _states_series, _labels_series = sess.run(
            [total_loss, train_step, current_state, logits_series, states_series, labels_series],
            feed_dict={
                batchX_placeholder:batchX,
                batchY_placeholder:batchY,
                init_state:_current_state
            })

            #print("Pred: ",_labels_series)
            #print()
            #print("logits: ",_logits_series)
            #print()
            #print("loss: ",_total_loss)
            #print()
            #print("states: ",_states_series)
            #print()

        loss_list.append(_total_loss)

        if epoch_idx%100 == 99:
            print("Step",epoch_idx*batch_size, "Loss", _total_loss)
            #print("Pred: ",batchY)
            #print()
            #print("logits: ",_logits_series)
            #print()
            plot(loss_list, _logits_series, batchX, batchY)

plt.ioff()
plt.show()