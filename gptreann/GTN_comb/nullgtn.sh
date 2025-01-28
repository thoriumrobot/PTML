#!/bin/bash

python main.py --dataset Null --model FastGTN --num_layers 3 --epoch 200 --lr 0.05 --channel_agg mean --num_channels 2 --non_local_weight -1 --K 1 --non_local
