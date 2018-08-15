#!/bin/bash

deal_output_dir="./output/"
deal_output_file="deals_and_auctions.txt"
display_hands=false
nplayers=4
ndeals=1000
max_auction_length=15
if [ "$nplayers" = 4 ] ; then
	max_auction_length=3
fi
iters=5

echo "dealing hands and bidding"

if [ "$display_hands" = true ] ; then
	python3 ./deal"$nplayers".py $ndeals | tee "$deal_output_dir""$deal_output_file"
else
	python3 ./deal"$nplayers".py $ndeals > "$deal_output_dir""$deal_output_file"
fi

python3 ./enc"$nplayers".py "$deal_output_dir""$deal_output_file" "$max_auction_length"

max_bid=$(($max_auction_length-1))
for ply in $(seq 0 $max_bid)
	do
		echo "training bidder for bid number" "$ply"
		python3 ./train.py "$nplayers" "$ply" "$iters"
	done
 
