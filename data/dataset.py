"""Defines the parking slot dataset for directional marking point detection."""
import json
import os
import os.path
import cv2 as cv
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from data.struct import MarkingPoint
from data.data_augmentation import HSVAdjust, ScaleCropPadAugmentation, Compose
# Augment: image [8,3,512,512] -> [8,6,16,16]
class ParkingSlotDataset(Dataset):
    """Parking slot dataset."""
    def __init__(self, root,mode='train',image_size=512, is_training=True):
        super(ParkingSlotDataset, self).__init__()
        self.root = root
        self.sample_names = []
        self.image_transform = ToTensor()
        # add from yolo v2 idea 
        self.image_size = image_size 
        self.is_training = is_training
        for file in os.listdir(root):
            if file.endswith(".json"):
                self.sample_names.append(os.path.splitext(file)[0])

    def __getitem__(self, index):
        """
        marking_points: list of points: [(x1,y1,theta1,label1), (x2,y2,theta2,label2)]
        """
        name = self.sample_names[index]
        image = cv.imread(os.path.join(self.root, name+'.jpg'))
        marking_points = []
        with open(os.path.join(self.root, name + '.json'), 'r') as file:
            for label in json.load(file):
                marking_points.append(MarkingPoint(*label))
        if self.is_training: 
            transformations = Compose([ScaleCropPadAugmentation()])
        # else: 
            # transformations = Compose([Resize(self.image_size)])
        image, marking_points = transformations((image, marking_points)) 

        image = self.image_transform(image)
        return image, marking_points

    def __len__(self):
        return len(self.sample_names)


# Original: image [8,3,600,600] -> prediction [8,6,18,18]

# class ParkingSlotDataset(Dataset):
#     """Parking slot dataset."""
#     def __init__(self, root):
#         super(ParkingSlotDataset, self).__init__()
#         self.root = root
#         self.sample_names = []
#         self.image_transform = ToTensor()
#         for file in os.listdir(root):
#             if file.endswith(".json"):
#                 self.sample_names.append(os.path.splitext(file)[0])

#     def __getitem__(self, index):
#         name = self.sample_names[index]
#         image = cv.imread(os.path.join(self.root, name+'.jpg'))
#         image = self.image_transform(image)
#         marking_points = []
#         with open(os.path.join(self.root, name + '.json'), 'r') as file:
#             for label in json.load(file):
#                 marking_points.append(MarkingPoint(*label))
#         return image, marking_points

#     def __len__(self):
#         return len(self.sample_names)


