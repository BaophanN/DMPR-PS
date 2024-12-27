"""
@author: Viet Nguyen <nhviet1009@gmail.com>
Modify by: baophan 
"""
import json 

import numpy as np
from random import uniform
import cv2
from data.struct import MarkingPoint

import math
class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for function_ in self.transforms:
            data = function_(data)
        return data


class Crop(object):

    def __init__(self, max_crop=0.1):
        super().__init__()
        self.max_crop = max_crop

    def __call__(self, data):
        image, marking_points = data
        h, width = image.shape[:2]
        # Generate random crop values for each side 
        cropped_left = uniform(0, self.max_crop)
        cropped_right = uniform(0, self.max_crop)
        cropped_top = uniform(0, self.max_crop)
        cropped_bottom = uniform(0, self.max_crop)
        # calculate new boundaries for cropping 
        new_xmin = int(cropped_left * width)
        new_ymin = int(cropped_top * h)
        new_xmax = int(width - cropped_right * width)
        new_ymax = int(h - cropped_bottom * h)
        # Crop the image 
        image = image[new_ymin:new_ymax, new_xmin:new_xmax, :]
        new_height, new_width = image.shape[:2]
        updated_marking_points = []
        for point in marking_points:
            x,y,theta, confidence = point 
            # Denormalize coordinates to get pixel value
            x_pixel = x * width 
            y_pixel = y * h
            # Check if the point lies within the new cropped region 
            if new_xmin <= x_pixel <= new_xmax and new_ymin <= y_pixel <= new_ymax: 
                x_new = (x_pixel - new_xmin) / new_width
                y_new = (y_pixel - new_ymin) / new_height 
 
                # Angle remains the same
                updated_marking_points.append([x_new, y_new, theta, confidence]) 



        return image, updated_marking_points


class VerticalFlip(object):
    # got a bug with T shape marking points, annotations contains only angle, flip operation cannot be replaced by adding angle 

    def __init__(self, prob=0):
        super().__init__()
        self.prob = prob

    def __call__(self, data):
        image, marking_points = data
        if uniform(0, 1) >= self.prob:
            image = cv2.flip(image, 1)
            updated_marking_points = []
            """
            x,y,theta,confidence -> x,1-y,-theta, confidence
            """
            for point in marking_points:
                # no flip -> wrong orientation
                if point[2] < 0: 
                    point[2] = -math.pi - point[2] 
                else: 
                    point[2] = math.pi - point[2] 
                updated_marking_points.append([1-point[0],point[1],point[2],point[3]])
            
        return image, updated_marking_points


class HSVAdjust(object):

    def __init__(self, hue=30, saturation=1.5, value=1.5, prob=0.5):
        super().__init__()
        self.hue = hue
        self.saturation = saturation
        self.value = value
        self.prob = prob

    def __call__(self, data):
        # stay the same 

        def clip_hue(hue_channel):
            hue_channel[hue_channel >= 360] -= 360
            hue_channel[hue_channel < 0] += 360
            return hue_channel

        image, marking_points = data
        adjust_hue = uniform(-self.hue, self.hue)
        adjust_saturation = uniform(1, self.saturation)
        if uniform(0, 1) >= self.prob:
            adjust_saturation = 1 / adjust_saturation
        adjust_value = uniform(1, self.value)
        if uniform(0, 1) >= self.prob:
            adjust_value = 1 / adjust_value
        image = image.astype(np.float32) / 255
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image[:, :, 0] += adjust_hue
        image[:, :, 0] = clip_hue(image[:, :, 0])
        image[:, :, 1] = np.clip(adjust_saturation * image[:, :, 1], 0.0, 1.0)
        image[:, :, 2] = np.clip(adjust_value * image[:, :, 2], 0.0, 1.0)

        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        image = (image * 255).astype(np.float32)

        return image, marking_points


class RandomScaleResizePadding(object):

    def __init__(self, fixed_size, min_scale=0.5, max_scale=1.5):
        super().__init__()
        self.fixed_size = fixed_size
        self.min_scale = min_scale 
        self.max_scale = max_scale 

    def __call__(self, data):
        image, marking_points = data
        h, width = image.shape[:2]
        scale = uniform(self.min_scale,self.max_scale)
        new_width = int(width * scale) 
        new_height = int(h * scale)
        # resize image with random scale factor 
        resized_image = cv2.resize(image, (new_width, new_height))
        # pad image to fixed size 
        padded_image = np.full((self.fixed_size,self.fixed_size,3),128,dtype=np.uint8)
        x_offset = (self.fixed_size - new_width) // 2 if new_width < self.fixed_size else 0 
        y_offset = (self.fixed_size - new_height) // 2 if new_height < self.fixed_size else 0

        padded_image[y_offset:y_offset+new_height,x_offset:x_offset+new_width] = resized_image
        return padded_image, marking_points

class ScaleCropPadAugmentation:
    def __init__(self, fixed_size=512, min_scale=0.5, max_scale=2.0, max_crop=0.2, pad_value=128):
        """
        Augmentation to handle scaling, cropping, and padding for square images.
        Args:
            fixed_size (int): Fixed size to which the final image is adjusted.
            min_scale (float): Minimum scale factor for random scaling.
            max_scale (float): Maximum scale factor for random scaling.
            max_crop (float): Maximum crop ratio for random cropping.
            pad_value (int): Value used for padding.
        """
        self.fixed_size = fixed_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_crop = max_crop
        self.pad_value = pad_value

    def __call__(self, data):
        image, marking_points = data

        # Step 1: Random Scaling
        image, marking_points = self.random_scale(image, marking_points)

        # Step 2: Random Cropping
        image, marking_points = self.random_crop(image, marking_points)

        # Step 3: Adjust to fixed size (Pad if smaller)
        image, marking_points = self.pad_to_fixed_size(image, marking_points)

        return image, marking_points

    def random_scale(self, image, marking_points):
        """Applies random scaling to the square image and adjusts marking points."""
        height, width = image.shape[:2]
        assert height == width, "Input image must be square."

        scale = uniform(self.min_scale, self.max_scale)
        new_size = int(width * scale)

        # Resize the image
        resized_image = cv2.resize(image, (new_size, new_size))

        # Update marking points
        updated_marking_points = []
        for point in marking_points:
            x, y, theta, confidence = point
            # Convert normalized coordinates to pixel values
            x_pixel = x * width
            y_pixel = y * height
            # Apply scaling
            x_pixel *= scale
            y_pixel *= scale
            # Normalize back to [0, 1]
            x_new = x_pixel / new_size
            y_new = y_pixel / new_size
            new_point = MarkingPoint(
                x=x_new, 
                y=y_new, 
                direction=theta, 
                shape=confidence 
            )
            updated_marking_points.append(new_point)

        return resized_image, updated_marking_points

    def random_crop(self, image, marking_points):
        """Applies random cropping to the square image and adjusts marking points."""
        height, width = image.shape[:2]
        assert height == width, "Input image must be square."

        # Generate random crop values
        crop_amount = int(uniform(0, self.max_crop) * width)
        new_size = width - crop_amount

        # Crop the image
        cropped_image = image[0:new_size, 0:new_size]

        # Update marking points
        updated_marking_points = []
        for point in marking_points:
            x, y, theta, confidence = point
            # Convert normalized coordinates to pixel values
            x_pixel = x * width
            y_pixel = y * height
            # Check if the point is within the cropped region
            if 0 <= x_pixel < new_size and 0 <= y_pixel < new_size:
                # Normalize back to [0, 1]
                x_new = x_pixel / new_size
                y_new = y_pixel / new_size
                new_point = MarkingPoint(
                    x=x_new, 
                    y=y_new, 
                    direction=theta, 
                    shape=confidence 
                )
                updated_marking_points.append(new_point)

        return cropped_image, updated_marking_points

    def pad_to_fixed_size(self, image, marking_points):
        """
        Adjust the image to the fixed size by resizing if larger and padding if smaller.
        Ensures marking points are correctly normalized after transformations.
        Args:
            image: Input image (height and width are equal).
            marking_points: List of marking points with normalized coordinates.
        Returns:
            The adjusted image and updated marking points.
        """
        height, width = image.shape[:2]
        assert height == width, "Input image must be square."

        # Case 1: Resize if the image is larger than the fixed size
        if width > self.fixed_size:
            scale = self.fixed_size / width
            resized_image = cv2.resize(image, (self.fixed_size, self.fixed_size))

            # Update marking points
            updated_marking_points = []
            for point in marking_points:
                x, y, theta, confidence = point
                # Convert normalized coordinates to pixel values
                x_pixel = x * width
                y_pixel = y * height
                # Apply resizing
                x_pixel *= scale
                y_pixel *= scale
                # Normalize back to [0, 1]
                x_new = x_pixel / self.fixed_size
                y_new = y_pixel / self.fixed_size
                new_point = MarkingPoint(
                    x=x_new, 
                    y=y_new, 
                    direction=theta, 
                    shape=confidence 
                )
                updated_marking_points.append(new_point)

            return resized_image, updated_marking_points

        # Case 2: Pad if the image is smaller than the fixed size
        elif width < self.fixed_size:
            # Create a new padded image
            padded_image = np.full((self.fixed_size, self.fixed_size, 3), self.pad_value, dtype=np.uint8)

            # Calculate offsets to center the image
            offset = (self.fixed_size - width) // 2
            padded_image[offset:offset+width, offset:offset+width] = image

            # Update marking points
            updated_marking_points = []
            for point in marking_points:
                x, y, theta, confidence = point
                # Convert normalized coordinates to pixel values
                x_pixel = x * width + offset
                y_pixel = y * height + offset
                # Normalize back to [0, 1]
                x_new = x_pixel / self.fixed_size
                y_new = y_pixel / self.fixed_size
                new_point = MarkingPoint(
                    x=x_new, 
                    y=y_new, 
                    direction=theta, 
                    shape=confidence 
                )
                updated_marking_points.append(new_point)

            return padded_image, updated_marking_points

        # Case 3: If the image is already the fixed size
        else:
            return image, marking_points

