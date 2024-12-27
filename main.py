import numpy as np 
import cv2 as cv
import json
from data.data_augmentation import *
import math

def plot_points(image, pred_points):
    """Plot marking points on the image."""
    if not pred_points:
        return
    height = image.shape[0]
    width = image.shape[1]
    for marking_point in pred_points:
        p0_x = width * marking_point[0] - 0.5
        p0_y = height * marking_point[1] - 0.5
        cos_val = math.cos(marking_point[2])
        sin_val = math.sin(marking_point[2])
        p1_x = p0_x + 50*cos_val
        p1_y = p0_y + 50*sin_val
        p2_x = p0_x - 50*sin_val
        p2_y = p0_y + 50*cos_val
        p3_x = p0_x + 50*sin_val
        p3_y = p0_y - 50*cos_val
        p0_x = int(round(p0_x))
        p0_y = int(round(p0_y))
        p1_x = int(round(p1_x))
        p1_y = int(round(p1_y))
        p2_x = int(round(p2_x))
        p2_y = int(round(p2_y))
        cv.line(image, (p0_x, p0_y), (p1_x, p1_y), (0, 0, 255), 2)
        cv.putText(image, str(1), (p0_x, p0_y),
                   cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        if marking_point[3] > 0.5:
            cv.line(image, (p0_x, p0_y), (p2_x, p2_y), (0, 0, 255), 2)
        else:
            p3_x = int(round(p3_x))
            p3_y = int(round(p3_y))
            cv.line(image, (p2_x, p2_y), (p3_x, p3_y), (0, 0, 255), 2)


if __name__ == "__main__":
    name = '0002'
    name = '20160725-3-1_20'
    root = 'datasets/annotations/train'
    image1 = cv.imread(f'{root}/{name}.jpg')
    image2 = cv.imread(f'{root}/{name}.jpg')        

    with open(f'{root}/{name}.json', 'r') as f:
        marking_points = json.load(f)  # load json label 
    # marking_points = json.load(f'{root}/{name}.json') # load json label 

        # if self.is_training: 
        #     transformations = Compose([HSVAdjust(), VerticalFlip(), Crop(), Resize(self.image_size)])
        # else: 
        #     transformations = Compose([Resize(self.image_size)])
        # image, marking_points = transformations((images, marking_points)) 
    plot_points(image1, marking_points)
    cv.imwrite(f'./sample_image_output/gt_{name}.jpg', image1, [int(cv.IMWRITE_JPEG_QUALITY), 100])
    transformations = Compose([HSVAdjust()])
    image2, marking_points = transformations((image2, marking_points)) 
    print('after transform', marking_points)
    plot_points(image2, marking_points)
    cv.imwrite(f'./sample_image_output/gt__transform_{name}.jpg', image2, [int(cv.IMWRITE_JPEG_QUALITY), 100])



