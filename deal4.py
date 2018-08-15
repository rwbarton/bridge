#!/usr/bin/env python3

from typing import List, Tuple
import os
import random
import re
import subprocess

count = 10 # number of hands
gib_dir = 'gib_dir'
temp_dir = 'tmp2'

Card = Tuple[int,str]

one_suit = list("AKQJT98765432")
deck = [(suit, card) for suit in [0,1,2,3] for card in one_suit]

def card_key(c: str) -> int:
    return one_suit.index(c)

def hand_to_str(cards: List[Card]) -> str:
    def suit_string(suit: int) -> str:
        suit_cards = [card for (suit1, card) in cards if suit1 == suit]
        suit_cards.sort(key=card_key)
        return ''.join(suit_cards)
    hand = '.'.join(suit_string(suit) for suit in [0,1,2,3])
    return hand

def deal_four_hands() -> Tuple[str,str,str,str]:
    shuffled_deck = deck[:]
    random.shuffle(shuffled_deck)
    north = shuffled_deck[0:13]
    east = shuffled_deck[13:26]
    south = shuffled_deck[26:39]
    west = shuffled_deck[39:52]
    return hand_to_str(north), hand_to_str(east), hand_to_str(south), hand_to_str(west)

def write_gib_file(filename: str, hand: str) -> None:
    f = open(filename, 'w')
    f.write(hand + '\n')
    f.write('n\n')              # North to deal
    f.write('-\n')              # none vulnerable
    f.close()

def build_inputs(directory: str, count: int) -> List[Tuple[str,str,str,str]]:
    os.makedirs(directory, exist_ok=True)
    filenames = [os.path.join(directory, 'hand%d' % i)
                 for i in range(0, count)]
    inNf = open(os.path.join(directory, 'inN'), 'w')
    inEf = open(os.path.join(directory, 'inE'), 'w')
    inSf = open(os.path.join(directory, 'inS'), 'w')
    inWf = open(os.path.join(directory, 'inW'), 'w')
    hands = []
    for f in filenames:
        north, east, south, west = deal_four_hands()
        hands.append((north, east, south, west))
        write_gib_file(f + 'N', north)
        inNf.write(os.path.abspath(f + 'N') + '\n')
        write_gib_file(f + 'E', east)
        inEf.write(os.path.abspath(f + 'E') + '\n')
        write_gib_file(f + 'S', south)
        inSf.write(os.path.abspath(f + 'S') + '\n')
        write_gib_file(f + 'W', west)
        inWf.write(os.path.abspath(f + 'W') + '\n')
    inNf.close()
    inEf.close()
    inSf.close()
    inWf.close()
    return hands

def invoke_gib(seat: str, filename: str) -> subprocess.Popen:
    # -Q: skip the lead/play phase.
    # This lets us avoid whichever GIB is on lead thinking about what to lead,
    # which is slow and apparently cannot be made faster, and also simplifies
    # our logic (don't need to tell GIB to advance to the next hand explicitly).
    return subprocess.Popen(['wine', 'bridge.exe', '-c', '-j', '-I', seat, '-T', '1', '-Q',
                             '-i', filename],
                            cwd=gib_dir, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)

def run_gib(filename: str, hands: List[Tuple[str,str,str,str]]) -> List[List[str]]:
    # GIB must run from its own directory (or the locations of its
    # data files could be specified on the command line)
    gibN = invoke_gib('-N', filename + 'N')
    gibE = invoke_gib('-E', filename + 'E')
    gibS = invoke_gib('-S', filename + 'S')
    gibW = invoke_gib('-W', filename + 'W')

    auctions = []  # type: List[List[bytes]]
    last_hand = None  # type: Tuple[str,str,str,str]
    for hand in hands:
        auction = []  # type: List[bytes]
        engines = [gibN, gibE, gibS, gibW]  # North deals
        while True:
            # send enqueued bids
            current = engines[0]

            # get bid from current engine
            while True:
                line = current.stdout.readline()
                #print('<', 0, line, flush=True)
                if line.startswith(b'I bid '):
                    m = re.match(b'^I bid ([1-7A-Z]+) as', line)
                    bid = m.group(1)
                    break

            # send bid to other engines
            for i in range(1,4):
                #print('>', i, bid, flush=True)
                engines[i].stdin.write(b'%s\n' % bid)
                engines[i].stdin.flush()

            # record bid
            auction.append(bid.decode('utf-8'))
            if auction[-3:] == ['P', 'P', 'P'] and len(auction) >= 4:
                break
            start = False

            engines = engines[1:] + engines[0:1]
            hand = hand[1:] + hand[0:1]

        auctions.append(auction)
        last_hand = hand
    return auctions

hands = build_inputs(temp_dir, count)
auctions = run_gib(os.path.abspath(os.path.join(temp_dir, 'in')), hands)
for hand, auction in zip(hands, auctions):
    print("%s\t%s" % (hand, auction))
