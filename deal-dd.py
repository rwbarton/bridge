#!/usr/bin/env python3

from typing import List, Tuple
import os
import random
import re
import subprocess

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

def deal_four_hands() -> List[str]:
    shuffled_deck = deck[:]
    random.shuffle(shuffled_deck)
    return [hand_to_str(shuffled_deck[13*i:13*(i+1)]) for i in [0,1,2,3]]

def write_gib_file(filename: str, hands: List[str], strain: str) -> None:
    f = open(filename, 'w')
    for seat, hand in zip(list("wnes"), hands):
        f.write(seat + ' ' + hand + '\n')
    f.write('w ' + strain + '\n') # West to lead
    f.close()

def build_inputs(directory: str, count: int) -> List[List[str]]:
    os.mkdir(directory)
    filenames = [os.path.join(directory, 'hand%d' % i)
                 for i in range(0, count)]
    inf = open(os.path.join(directory, 'in'), 'w')
    deals = []
    for f in filenames:
        hands = deal_four_hands()
        deals.append(hands)
        write_gib_file(f, hands, 'n')
        inf.write(os.path.abspath(f) + '\n')
        # suits: all or just spades?
    inf.close()
    return deals

gib_dir = '/home/rwbarton/.wine/drive_c/Bridge Base Online/robots/gib'

def run_gib(filename: str) -> List[int]:
    # GIB must run from its own directory (or the locations of its
    # data files could be specified on the command line)
    result = subprocess.run(['wine', 'bridge.exe', '-j', '-d', '-i', filename],
                            cwd=gib_dir,
                            stdout=subprocess.PIPE)

    trick_counts = []  # type: List[int]
    for l in result.stdout.splitlines():
        l = l.rstrip()
        if len(l) == 1:
            trick_counts.append(int(l.decode('utf-8'), 16))

    return trick_counts

deals = build_inputs('tmpDD', 1000)
counts = run_gib(os.path.abspath('tmpDD/in'))
for deal, count in zip(deals, counts):
    print('%s\t%d' % (' '.join(deal), count))
