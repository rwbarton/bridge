import itertools
import numpy as np
import tensorflow as tf

one_suit = list("AKQJT98765432")

def vec_to_hand(hand) -> str:
    return '.'.join(''.join(one_suit[i] for i in range(0,13) if hand[13*s+i])
                    for s in range(0,4))

def idx_to_bid(idx: int) -> str:
    if idx == 0:
        return 'P'
    if idx == 1:
        return 'X'
    if idx == 2:
        return 'XX'
    return str(int((idx - 3) / 5 + 1)) + "CDHSN"[(idx - 3) % 5]

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def load_bidder(ply: int):
    INPUTS = 52 + 38 * ply
    MIDDLE = 500
    x = tf.placeholder(tf.float32, [None, INPUTS])
    W1 = weight_variable([INPUTS, MIDDLE])
    b1 = bias_variable([MIDDLE])
    x1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    W = weight_variable([MIDDLE,38])
    b = bias_variable([38])
    y = tf.matmul(x1, W) + b
    y_ = tf.placeholder(tf.float32, [None, 38])

    prediction = tf.argmax(y,1)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    saver = tf.train.Saver({"Variable": W1, "Variable_1": b1,
                            "Variable_2": W, "Variable_3": b})
    saver.restore(sess, "models/hands%d_10.ckpt" % (ply,))
    
    def bid(auction, hand):
        assert len(auction) == ply
        xin = np.hstack([hand] + list(np.eye(38, dtype=np.float32)[i] for i in auction))
        pred = sess.run(prediction, feed_dict={x: np.stack([xin])})[0]
        return pred
    
    return bid

handsS = np.load('dat/handsS.npy')
handsN = np.load('dat/handsN.npy')
testing_hands_start = int(0.8 * handsS.shape[0])
handsS = handsS[testing_hands_start:]
handsN = handsN[testing_hands_start:]

bidders = []
for i in range(12):
    bidders.append(load_bidder(i))

gib_output = open('4M2.txt', 'r')

for _ in range(testing_hands_start):
    gib_output.readline()
print("done")

line = 0
for hS, hN in zip(handsS, handsN):
    auction = []
    while len(auction) < 2 or auction[-1] != 0:
        ply = len(auction)
        next_bid = bidders[ply](auction, [hS, hN][ply % 2]) if ply < 12 else 0
        auction.append(next_bid)
    contract = auction[-2]
    #print(list(idx_to_bid(b) for b in auction))
    gib_line = gib_output.readline().rstrip()
    gib_auction = eval(gib_line.split('\t')[1])
    gib_contract = gib_auction[-2]
    if idx_to_bid(contract) != gib_contract:
        print(line, vec_to_hand(hN), vec_to_hand(hS), idx_to_bid(contract), gib_contract)
    line += 1

# for hS, hN in zip(handsS, handsN):
#     print(vec_to_hand(hS), vec_to_hand(hN))
