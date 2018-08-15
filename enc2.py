#!/usr/bin/env python3

from typing import List

import progressbar
import numpy as np
import sys
import argparse
import os

OUTPUT_DIR_HANDS = "dat"
OUTPUT_DIR_BIDS = os.path.join(OUTPUT_DIR_HANDS, "bids")

one_suit = list("AKQJT98765432")

def hand_to_vec(hand: str) -> List[int]:
    result = []
    for suit in hand.split('.'):
        for card in one_suit:
            result.append(1 if card in suit else 0)
    return result

# P X XX 1C 1D ... 7N
# Note: X and XX are untested
def bid_to_idx(bid: str) -> int:
    if bid == 'P':
        return 0
    if bid == 'X':
        return 1
    if bid == 'XX':
        return 2
    return 3 + 5 * (int(bid[0:1]) - 1) + "CDHSN".index(bid[1:2])

def idx_to_bid(idx: int) -> str:
    if idx == 0:
        return 'P'
    if idx == 1:
        return 'X'
    if idx == 2:
        return 'XX'
    return str(int((idx - 3) / 5 + 1)) + "CDHSN"[(idx - 3) % 5]

def bid_to_np(bid):
    return np.eye(38, dtype=np.float32)[bid_to_idx(bid)]

def main():
    #despite what tabbott says, we aren't doing this right now
    #parser = argparse.ArgumentParser()  
    #parser.parse_args() 

    if not len(sys.argv) > 2: raise AssertionError("Command line arguments are required to specify the input file name and max auction length.")
    input_file = sys.argv[1]
    try:
        max_auction_length = int(sys.argv[2])
    except ValueError:
        print("Max auction length must be an integer.")


    with open(input_file, 'r') as fin:
        ROWS = sum(1 for line in fin if line.rstrip())

    handsN = np.zeros((ROWS, 52), dtype=np.float32)
    handsS = np.zeros((ROWS, 52), dtype=np.float32)
    bids = [np.full(ROWS, -1, dtype=np.int8) for _ in range(15)]
    bar = progressbar.ProgressBar()

    f = open(input_file, 'r')
    for i in bar(range(ROWS)):
        l = f.readline()
        (handN, handS), auction = (eval(col) for col in l.rstrip().split('\t'))
        handsN[i] = np.array(hand_to_vec(handN), dtype=np.float32)
        handsS[i] = np.array(hand_to_vec(handS), dtype=np.float32)
        for j, b in enumerate(auction):
            bids[j][i] = bid_to_idx(b)

    os.makedirs(OUTPUT_DIR_HANDS, exist_ok=True)
    os.makedirs(OUTPUT_DIR_BIDS, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR_HANDS, 'handsN'), handsN)
    np.save(os.path.join(OUTPUT_DIR_HANDS, 'handsS'), handsS)
    for j in range(max_auction_length):
        np.save(os.path.join(OUTPUT_DIR_BIDS,'bids%d') % (j,), bids[j])

if __name__ == "__main__":
    main()
