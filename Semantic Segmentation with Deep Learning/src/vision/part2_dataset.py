#!/usr/bin/python3

import os
import os.path
import re
from typing import List, Tuple
import glob

import cv2
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset


"""
Dataset class for semantic segmentation data.
"""

def make_dataset(split: str, data_root: str, data_list_fpath: str) -> List[Tuple[str, str]]:
    """
    Create list of (image file path, label file path) pairs, as ordered in the
    data_list_fpath .txt file.

    Args:
        split: string representing split of data set to use, must be either
            'train','val','test'
        data_root: path to where data lives, and where relative image paths are
            relative to
        data_list_fpath: path to .txt file with relative image paths and their
            corresponding GT path

    Returns:
        image_label_list: list of 2-tuples, each 2-tuple is comprised of an absolute image path
            and an absolute label path
    """
    assert split in ["train", "val", "test"]

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    
    image_label_list = []
    
    # Read the data list file
    with open(data_list_fpath, 'r') as f:
        lines = f.readlines()
    
    # Process each line to create absolute paths
    for line in lines:
        line = line.strip()
        if line:  # Skip empty lines
            # Split the line into image path and label path
            parts = line.split()
            if len(parts) >= 2:
                relative_image_path = parts[0]
                relative_label_path = parts[1]
                
                # Create absolute paths by joining with data_root
                absolute_image_path = os.path.join(data_root, relative_image_path)
                absolute_label_path = os.path.join(data_root, relative_label_path)
                
                # Add the tuple to our list
                image_label_list.append((absolute_image_path, absolute_label_path))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    print(f"List of (image,label) pairs {split} list generated!")
    return image_label_list


class SemData(Dataset):
    def __init__(self, split: str, data_root: str, data_list_fpath: str, transform=None) -> None:
        """
        Dataloader class for semantic segmentation datasets.

        Args:
            split: string representing split of data set to use, must be either
                'train','val','test'
            data_root: path to where data lives, and where relative image paths
                are relative to
            data_list_fpath: path to .txt file with relative image paths
            transform: Pytorch torchvision transform
        """
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list_fpath)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the transformed RGB image and ground truth, as tensors.

        We will not load the image using PIL, since we will not be using the
        default Pytorch transforms.

        You can read in the image and label map using imageio or opencv, but
        the transform should accept a (H,W,C) float 32 RGB image (not BGR like
        OpenCV reads), and a (H,W) int64 label map.

        Args:
            index: index of the example to retrieve within the dataset

        Returns:
            image: tensor of shape (C,H,W), with type torch.float32
            label: tensor of shape (H,W), with type torch.long (64-bit integer)
        """

        ###########################################################################
        # TODO: YOUR CODE HERE                                                    #
        ###########################################################################
        
        # Get the image and label paths for this index
        image_path, label_path = self.data_list[index]
        
        # Load the image using imageio (to get RGB format)
        image = imageio.imread(image_path)
        
        # Convert to float32 as required
        image = image.astype(np.float32)
        
        # Load the label using imageio
        label = imageio.imread(label_path)
        
        # Ensure label is 2D (H,W) and convert to int64
        if len(label.shape) == 3:
            # If label is RGB/RGBA, take only one channel (assuming all channels are the same)
            label = label[:, :, 0]
        label = label.astype(np.int64)
        
        # Apply transform if provided
        if self.transform:
            image, label = self.transform(image, label)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        return image, label


def get_label_paths(label_path):
    """
    Args:
        label_path: path to where data lives, and where relative train/test paths
            are relative to
    Returns:
        label_paths: dictionary which contains {image0 : path to corresponding label, ...}
        i.e)
        {'um_000080.png': 'data_root/kitti/training/gt_image_2/um_road_000080.png', ...}
    """
    label_paths = {re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
                   for path in glob.glob(os.path.join(label_path, '*_road_*.png'))}
    return label_paths

class KittiData(Dataset):
    """
    Dataloader class for kitti road segmentation datasets.
    Args:
        split: string which indicates train or test
        data_root: path to where data lives, and where relative train/test paths
            are relative to
        transform: Pytorch torchvision transform
    """
    def __init__(self, split:str, data_root: str, transform=None):
        """
        For convenience we are using train_path can be path to the training
        dataset or test dataset depending on the value of split variable.
        """
        self.transform = transform
        if split == "train":
            self.train_path = data_root + "/training/image_2"
            self.label_path = data_root + "/training/gt_image_2"
        else:
            self.train_path = data_root + "/testing/image_2"
            self.label_path = data_root + "/testing/gt_image_2"

    def __len__(self):
        path, dirs, files = next(os.walk(self.train_path))
        return len(files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the transformed RGB image and ground truth, as tensors.
        We will not load the image using PIL, since we will not be using the
        default Pytorch transforms.
        You can read in the image and label map using imageio or opencv, but
        the transform should accept a (H,W,C) float 32 RGB image (not BGR like
        OpenCV reads), and a (H,W) int64 label map.
        Args:
            index: index of the example to retrieve within the dataset
        Returns:
            image: tensor of shape (C,H,W), with type torch.float32
            label: tensor of shape (H,W), with type torch.long (64-bit integer)

        Resize the image and label so that H=256, W=256. Consider using cv2.resize()
        """

        label_paths = get_label_paths(self.label_path)
        image_path = list(label_paths)[index]

        image = cv2.imread(os.path.join(self.train_path, image_path), cv2.IMREAD_COLOR) # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = cv2.resize(image, (256, 256))
        image = np.float32(image)

        label_path = label_paths[image_path]
        label = imageio.imread(label_path)
        label = cv2.resize(label, (256, 256))
        label = label[:,:,2]
        truth_table = label == 255
        label = np.invert(truth_table)
        label = label.astype(np.int64)

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))

        if self.transform:
            image, label = self.transform(image, label)

        return image, label
