#!/usr/bin/env python3

import sys
import progressbar
import numpy as np
import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


[ply, iters, learn] = [int(a) for a in sys.argv[1:]]

hands = np.load('dat/hands%s.npy' % ("SN"[ply % 2],))
bids = []
for p in range(ply+1):
    bids.append(np.load('dat/bids%d.npy' % (p,)))
print('Loaded')
# bids[ply] is the output we are trying to learn,
# previous bids (and hands) are the input
# Only interested in hands where we are faced with a decision,
# i.e., where bids[ply] is not -1
valid = bids[ply] != -1
hands = hands[valid]            # not sure why, but seems to work
for p in range(ply+1):
    bids[p] = bids[p][valid]
print('Filtered')
# Now build the actual network input/output
total_hands = bids[0].size
#total_hands //= 40
print(total_hands, 'hands')
INPUTS = 52 + 38 * ply          # our hand + preceding calls
all_in = np.zeros((total_hands, INPUTS), dtype=np.float32)
all_out = np.zeros((total_hands, 38), dtype=np.float32)
eye = np.eye(38, dtype=np.float32)
bar = progressbar.ProgressBar()
for i in bar(range(total_hands)):
    all_in[i][0:52] = hands[i]
    for p in range(ply):
        all_in[i][(52+38*p):(52+38*(p+1))] = eye[bids[p][i]]
    all_out[i] = eye[bids[ply][i]]
# Free things we no longer need
del hands
del bids
del valid

MIDDLE = 500
x = tf.placeholder(tf.float32, [None, INPUTS])
W1 = weight_variable([INPUTS, MIDDLE])
b1 = bias_variable([MIDDLE])
x1 = tf.nn.relu(tf.matmul(x, W1) + b1)
W = weight_variable([MIDDLE,38])
b = bias_variable([38])
y = tf.matmul(x1, W) + b
y_ = tf.placeholder(tf.float32, [None, 38])

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

#train_step = tf.train.GradientDescentOptimizer(0.015).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(learn/10000).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

BATCH_SIZE=20
training_hands = int(0.8*total_hands)
for l in range(iters):
    tot = 0
    bar = progressbar.ProgressBar()
    for k in bar(range(training_hands//BATCH_SIZE)):
        batch_xs = all_in[BATCH_SIZE*k:BATCH_SIZE*(k+1)]
        batch_ys = all_out[BATCH_SIZE*k:BATCH_SIZE*(k+1)]
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        # score = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
        # tot += score
        # if k % 200 == 0:
        #     print(k * BATCH_SIZE, tot/200)
        #     tot = 0

    # Measure accuracy
    # Doing this in a single batch uses an unreasonable amount of memory.
    ACC_BATCH_SIZE=10000
    next_i = training_hands
    correct = 0
    while next_i < total_hands:
        i = next_i
        next_i += ACC_BATCH_SIZE
        correct += sess.run(accuracy, feed_dict={x: all_in[i:next_i],
                                                 y_: all_out[i:next_i]})
    acc = correct / (total_hands - training_hands)
    print(l, "Acc", acc, flush=True)

saver = tf.train.Saver()
save_path = saver.save(sess, 'models/hands%d_%d.ckpt' % (ply, iters))
print("Model saved in %s" % save_path)
