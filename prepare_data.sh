#!/bin/bash
LABEL_DIRECTORY=/home/baogp4/DMPR-PS/datasets/ps_json_label/training
IMAGE_DIRECTORY=/home/baogp4/DMPR-PS/datasets/training 
OUTPUT_DIRECTORY=/home/baogp4/DMPR-PS/datasets/annotations
python prepare_dataset.py --dataset trainval --label_directory ${LABEL_DIRECTORY} --image_directory ${IMAGE_DIRECTORY} --output_directory ${OUTPUT_DIRECTORY}
# python prepare_dataset.py --dataset test --label_directory $LABEL_DIRECTORY --image_directory $IMAGE_DIRECTORY --output_directory $OUTPUT_DIRECTORY