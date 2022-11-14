#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1
python3  -u main.py --dataset=$1 --optimizer='nfedprox-Copy3'  \
            --learning_rate=0.01 --num_rounds=25 --clients_per_round=$2 \
            --eval_every=1 --batch_size=10 \
            --num_epochs=$3 \
            --model=$9 \
            --drop_percent=0 \
            --mu=$4 --L=$5 --Clip=$6 --epsilon=$7 --delta=$8\
