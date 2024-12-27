#!/bin/bash
# Number of GPUs to use
NUM_GPUS=1
CUDA_VISIBLE_DEVICES=0
TRAIN_DIRECTORY=~/DMPR-PS/datasets/annotations/train


python train.py \
           --dataset_directory $TRAIN_DIRECTORY \
           --gpu_id 0 \
           --num_epochs 1 \
           --batch_size 16 \
           --depth_factor 32 \
           --data_loading_workers 4 \
           --lr 0.0001 \
           --enable_visdom
