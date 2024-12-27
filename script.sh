#!/bin/bash
DETECTOR_WEIGHTS=./model/dmpr_pretrained_weights.pth
TEST_DIRECTORY=./datasets/parking_slot/testing/all
LABEL_DIRECTORY=./datasets/parking_slot/ps_json_label/testing/all
IMAGE_DIRECTORY=./datasets/parking_slot/testing/all 
python inference.py --mode image --detector_weights ./model/dmpr_pretrained_weights.pth --inference_slot

python evaluate.py --dataset_directory ./datasets/parking_slot/ps_json_label/testing/all --detector_weights ./model/dmpr_pretrained_weights.pth

python ps_evaluate.py --label_directory ./datasets/parking_slot/ps_json_label/testing/all --image_directory ./datasets/parking_slot/testing/all --detector_weights ./model/dmpr_pretrained_weights.pth
python inference.py --mode video --detector_weights ./model/dmpr_pretrained_weights.pth --video ./input.avi --inference_slot

python inference.py --mode video --detector_weights ./model/dmpr_pretrained_weights.pth --video ./sample_video_input/cropped_topview_1666775453007.mp4 --inference_slot --save --gpu_id 03
python inference.py --mode video --detector_weights ./weights/dp_detector_11.pth --video ./sample_video_input/topview_1666775453007.mp4 --inference_slot --save --gpu_id 0

# best case 
python inference.py --mode image --detector_weights ./weights/dp_detector_11.pth --inference_slot --save

# detect and collect threshold from prediction 
# train
python naive_train.py --dataset_directory datasets/parking_slot/annotations/train --gpu_ids 0,1,2,3 \
