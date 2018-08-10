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

def deal_two_hands() -> Tuple[str,str]:
    shuffled_deck = deck[:]
    random.shuffle(shuffled_deck)
    north = shuffled_deck[0:13]
    south = shuffled_deck[13:26]
    return hand_to_str(north), hand_to_str(south)

def write_gib_file(filename: str, hand: str) -> None:
    f = open(filename, 'w')
    f.write(hand + '\n')
    f.write('s\n')              # South to deal
    f.write('-\n')              # none vulnerable
    f.close()

def build_inputs(directory: str, count: int) -> List[Tuple[str,str]]:
    os.mkdir(directory)
    filenames = [os.path.join(directory, 'hand%d' % i)
                 for i in range(0, count)]
    inNf = open(os.path.join(directory, 'inN'), 'w')
    inSf = open(os.path.join(directory, 'inS'), 'w')
    hands = []
    for f in filenames:
        north, south = deal_two_hands()
        hands.append((north, south))
        write_gib_file(f + 'N', north)
        inNf.write(os.path.abspath(f + 'N') + '\n')
        write_gib_file(f + 'S', south)
        inSf.write(os.path.abspath(f + 'S') + '\n')
    inNf.close()
    inSf.close()
    return hands

def run_gib(filename: str, count: int) -> List[List[str]]:
    # GIB must run from its own directory (or the locations of its
    # data files could be specified on the command line)
    gibN = subprocess.Popen(['wine', 'bridge.exe', '-j', '-I', '-T', '1', '-N',
                             '-i', filename + 'N'],
                            cwd=gib_dir, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)
    gibS = subprocess.Popen(['wine', 'bridge.exe', '-j', '-I', '-T', '1', '-S',
                             '-i', filename + 'S'],
                            cwd=gib_dir, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)

    auctions = []  # type: List[List[bytes]]
    for _ in range(count):
        auction = []  # type: List[bytes]
        start = True            # first Pass does not end the auction
        engines = [gibS, gibN]  # South deals
        queued_bids = [[], []]
        while True:
            # send enqueued bids
            current = engines[0]

            for q in queued_bids[0]:
                #print('>', q, flush=True)
                current.stdin.write(b'%s\n' % q)
                current.stdin.flush()
            queued_bids[0] = []

            # get bid from current engine
            while True:
                line = current.stdout.readline()
                #print('<', line, flush=True)
                if line.startswith(b'I bid '):
                    m = re.match(b'^I bid ([1-7A-Z]+) as', line)
                    bid = m.group(1)
                    break

            # next opponent will pass
            queued_bids[0].append(b'P')
            queued_bids[1].append(bid)
            queued_bids[1].append(b'P')

            # record bid
            auction.append(bid.decode('utf-8'))
            if bid == b'P' and not start:
                break
            start = False

            engines = engines[1:] + engines[0:1]
            queued_bids = queued_bids[1:] + queued_bids[0:1]

        # auction is over; tell engines to advance to the next hand
        for engine in engines:
            engine.stdin.write(b'q\n')
            engine.stdin.flush()
            # it might be a good idea to skip ahead to the next
            # "Enter [NS] hand:" message
        auctions.append(auction)
    return auctions

hands = build_inputs(temp_dir, count)
auctions = run_gib(os.path.abspath(os.path.join(temp_dir, 'in')), count)
for hand, auction in zip(hands, auctions):
    print("%s\t%s" % (hand, auction))
    
