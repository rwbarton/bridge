#!/bin/bash

deal_output_dir="./output/"
deal_output_file="deals_and_auctions.txt"
display_hands=true
nhands=2

echo "dealing hands and bidding"

if [ "$display_hands" = true ] ; then
	python3 ./deal"$nhands".py | tee "$deal_output_dir""$deal_output_file"
else
	python3 ./deal"$nhands".py > "$deal_output_dir""$deal_output_file"
fi

python3 ./enc.py "$deal_output_dir""$deal_output_file"
