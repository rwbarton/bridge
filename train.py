#!/usr/bin/env python3

import sys
import argparse
import progressbar
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

HANDS_DIR = "dat"
BIDS_DIR = "dat/bids"
OUTPUT_DIR = "models"
NBIDS = 38
NCARDS = 52
learn = 0.001

parser = argparse.ArgumentParser(description='Train bidders.')
parser.add_argument('nplayers', metavar='N', type=int,
                    help='number of players')
parser.add_argument('ply', metavar='P', type=int,
                    help='number of bid')
parser.add_argument('iters', metavar='P', type=int,
                    help='number of learning iterations')
#parser.add_argument('learn', metavar='L', type=int,
#                    help='learning rate',default=0.001)
args = parser.parse_args()

nplayers = args.nplayers
ply = args.ply
iters = args.iters
if not nplayers in {2,4}:
    print("number of of players must be 2 or 4")
    sys.exit(2)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


if nplayers == 4:
    hands = np.load(os.path.join(HANDS_DIR, 'hands%s.npy' % ("NESW"[ply % 4],)))
else:
    hands = np.load(os.path.join(HANDS_DIR, 'hands%s.npy' % ("SN"[ply % 2],)))

bids = []
for p in range(ply+1):
    bids.append(np.load(os.path.join(BIDS_DIR,'bids%d.npy' % (p,))))
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
if total_hands == 0:
    print("Cannot train with 0 hands. Quitting.")
    sys.exit(0)

INPUTS = NCARDS + NBIDS * ply          # our hand + preceding calls
all_in = np.zeros((total_hands, INPUTS), dtype=np.float32)
all_out = np.zeros((total_hands, NBIDS), dtype=np.float32)
eye = np.eye(NBIDS, dtype=np.float32)
bar = progressbar.ProgressBar()
for i in bar(range(total_hands)):
    all_in[i][0:NCARDS] = hands[i]
    for p in range(ply):
        all_in[i][(NCARDS+NBIDS*p):(NCARDS+NBIDS*(p+1))] = eye[bids[p][i]]
    all_out[i] = eye[bids[ply][i]]
# Free things we no longer need
del hands
del bids
del valid

def train_tf_model():
    MIDDLE = 500
    x = tf.placeholder(tf.float32, [None, INPUTS])
    W1 = weight_variable([INPUTS, MIDDLE])
    b1 = bias_variable([MIDDLE])
    x1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    W = weight_variable([MIDDLE,NBIDS])
    b = bias_variable([NBIDS])
    y = tf.matmul(x1, W) + b
    y_ = tf.placeholder(tf.float32, [None, NBIDS])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    #train_step = tf.train.GradientDescentOptimizer(0.015).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(learn).minimize(cross_entropy)

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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = saver.save(sess, os.path.join(OUTPUT_DIR, 'hands%d_%d.ckpt' % (ply, iters)))
    print("Model saved in %s" % save_path)

def train_tf_keras_model():
    model = keras.Sequential()
    NUNITS = [500, 500]
    training_hands = int(0.8*total_hands)
    BATCH_SIZE=20

    for unit_count in NUNITS:
        model.add(keras.layers.Dense(unit_count, activation='relu'))
    model.add(keras.layers.Dense(NBIDS, activation='softmax'))
    model.compile(optimizer=tf.train.AdamOptimizer(learn),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    model.fit(all_in[:training_hands], all_out[:training_hands], epochs=iters, batch_size=BATCH_SIZE, validation_data = (all_in[training_hands:], all_out[training_hands:]))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, 'hands%d_%d.ckpt' % (ply, iters))
    model.save_weights(save_path)
    print("Model saved in %s" % save_path)

train_tf_keras_model()    