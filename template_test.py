"""Inference demo of directional point detector."""
import math
import cv2 as cv
import numpy as np
import torch
from torchvision.transforms import ToTensor
import config
from data import get_predicted_points, pair_marking_points, calc_point_squre_dist, pass_through_third_point
from model import DirectionalPointDetector
from util import Timer
from data.struct import MarkingPoint
while True:
    # shape < 0.5: T, >=0.5 L
    # Define templates globally to avoid redefining them in every function call
    templates = {}
    roi_size = 50
    template_size = roi_size * 2
    # T-Shape Template
    t_shape_template = np.zeros((template_size, template_size), dtype=np.float32)
    cv.line(t_shape_template, (roi_size, roi_size-50), (roi_size, roi_size+50), 255, 2)  # Horizontal line
    cv.line(t_shape_template, (roi_size, roi_size), (roi_size+50, roi_size), 255, 2)     
    templates['t_shape'] = t_shape_template / 255.0 # binarize

    # L-Shape Template
    l_shape_template = np.zeros((template_size, template_size), dtype=np.float32)
    cv.line(l_shape_template, (roi_size, roi_size), (roi_size + 50, roi_size), 255, 2)
    cv.line(l_shape_template, (roi_size, roi_size), (roi_size, roi_size - 50), 255, 2)
    templates['l_shape'] = l_shape_template / 255.0 # binarize 
    # cv.imshow('t',templates['t_shape'])
    cv.imshow('t',templates['t_shape'])

    cv.waitKey(1)
    # print('haha',templates['t_shape'])