"""
Script with Pytorch's dataloader class
"""

import glob
import os
from typing import Dict, List, Tuple

import torch
import torch.utils.data as data
import torchvision
from PIL import Image
import csv
import pandas as pd
import numpy as np


class ImageLoader(data.Dataset):
    """Class for data loading"""

    train_folder = "train"
    test_folder = "test"

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: torchvision.transforms.Compose = None,
    ):
        """Initialize the dataloader and set `curr_folder` for the corresponding data split.

        Args:
            root_dir: the dir path which contains the train and test folder
            split: 'test' or 'train' split
            transform: the composed transforms to be applied to the data
        """
        self.root = os.path.expanduser(root_dir)
        self.transform = transform
        self.split = split

        if split == "train":
            self.curr_folder = os.path.join(root_dir, self.train_folder)
        elif split == "test":
            self.curr_folder = os.path.join(root_dir, self.test_folder)

        self.class_dict = self.get_classes()
        self.dataset = self.load_imagepaths_with_labels(self.class_dict)

    def load_imagepaths_with_labels(
        self, class_labels: Dict[str, int]
    ) -> List[Tuple[str, int]]:
        """Fetches all (image path,label) pairs in the dataset.

        Args:
            class_labels: the class labels dictionary, with keys being the classes in this dataset
        Returns:
            list[(filepath, int)]: a list of filepaths and their class indices
        """

        img_paths = []  # a list of (filename, class index)

  

        for class_name, label in class_labels.items():
            class_dir = os.path.join(self.curr_folder, class_name)
            if not os.path.isdir(class_dir):
                continue

            pattern = os.path.join(class_dir, "*.jpg")
            files = glob.glob(pattern)
            files.sort()
            for fullpath in files:
                img_paths.append((fullpath, label))

  

        return img_paths

    def get_classes(self) -> Dict[str, int]:
        """Get the classes (which are folder names in self.curr_folder)

        NOTE: Please make sure that your classes are sorted in alphabetical order
        i.e. if your classes are ['apple', 'giraffe', 'elephant', 'cat'], the
        class labels dictionary should be:
        {"apple": 0, "cat": 1, "elephant": 2, "giraffe":3}

        If you fail to do so, you will most likely fail the accuracy
        tests on Gradescope

        Returns:
            Dict of class names (string) to integer labels
        """

        classes = dict()
  

        # List all directories in self.curr_folder and sort them alphabetically
        if not os.path.isdir(self.curr_folder):
            return classes

        entries = [d for d in os.listdir(self.curr_folder) if os.path.isdir(os.path.join(self.curr_folder, d))]
        entries.sort()

        for idx, cname in enumerate(entries):
            classes[cname] = idx

  
        return classes

    def load_img_from_path(self, path: str) -> Image:
        """Loads an image as grayscale (using Pillow).

        Note: do not normalize the image to [0,1]

        Args:
            path: the file path to where the image is located on disk
        Returns:
            image: grayscale image with values in [0,255] loaded using pillow
                Note: Use 'L' flag while converting using Pillow's function.
        """

        img = None
  

        # Open image using Pillow and convert to grayscale ('L'). Do not
        # normalize to [0,1] here; return the PIL Image with pixel values
        # in [0,255].
        with Image.open(path) as im:
            img = im.convert("L")

  
        return img

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Fetches the item (image, label) at a given index.

        Note: Do not forget to apply the transforms, if they exist

        Hint:
        1) get info from self.dataset
        2) use load_img_from_path
        3) apply transforms if valid

        Args:
            index: Index
        Returns:
            img: image of shape (H,W)
            class_idx: index of the ground truth class for this image
        """
        img = None
        class_idx = None

  

        # Get (path, label) from the dataset list
        path, class_idx = self.dataset[index]

        # Load image (PIL Image in 'L' mode)
        pil_img = self.load_img_from_path(path)

        # If transforms provided, apply them (they expect PIL Images)
        if self.transform is not None:
            img = self.transform(pil_img)
        else:
            # Convert PIL Image to torch tensor without normalizing to [0,1]
            # ensure channel-first shape (C,H,W) for grayscale images
            arr = np.asarray(pil_img)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]  # shape -> (1, H, W)
            img = torch.from_numpy(arr).float()

  
        return img, class_idx

    def __len__(self) -> int:
        """Returns the number of items in the dataset.

        Returns:
            l: length of the dataset
        """

        l = 0

  

        l = len(self.dataset)
        
  
        return l


class MultiLabelImageLoader(data.Dataset):
    """Class for data loading"""

    train_folder = "train"
    test_folder = "test"

    def __init__(
        self,
        root_dir: str,
        labels_csv: str,
        split: str = "train",
        transform: torchvision.transforms.Compose = None,
    ):
        """Initialize the dataloader and set `curr_folder` for the corresponding data split.

        Args:
            root_dir: the dir path which contains the train and test folder
            labels_csv: the path to the csv file containing ground truth labels
            split: 'test' or 'train' split
            transform: the composed transforms to be applied to the data
        """
        self.root = os.path.expanduser(root_dir)
        self.labels_csv = labels_csv
        self.transform = transform
        self.split = split

        if split == "train":
            self.curr_folder = os.path.join(root_dir, self.train_folder)
        elif split == "test":
            self.curr_folder = os.path.join(root_dir, self.test_folder)

        self.dataset = self.load_imagepaths_with_labels()

    def load_imagepaths_with_labels(self) -> List[Tuple[str, torch.Tensor]]:
        """Fetches all (image path,labels) pairs in the dataset from csv file. Ensure that only
        the images from the classes in ['coast', 'highway', 'mountain', 'opencountry', 'street']
        are included. 

        NOTE: Be mindful of the returned labels type

        Returns:
            list[(filepath, list(int))]: a list of filepaths and their labels
        """

        img_paths = []  # a list of (filename, class index)

  

        # CSV format: class,image_filename,<7 binary attributes>
        # We only include rows whose class is in the allowed set
        allowed = {"coast", "highway", "mountain", "opencountry", "street"}

        # Read CSV
        with open(self.labels_csv, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                cls, img_name = row[0], row[1]
                if cls not in allowed:
                    continue

                # remaining entries are attribute binaries; convert to int list
                labels = [int(x) for x in row[2:]]

                img_path = os.path.join(self.curr_folder, cls, img_name)
                img_paths.append((img_path, torch.tensor(labels, dtype=torch.float32)))

  

        return img_paths


    def load_img_from_path(self, path: str) -> Image:
        """Loads an image as grayscale (using Pillow).

        Note: do not normalize the image to [0,1]

        Args:
            path: the file path to where the image is located on disk
        Returns:
            image: grayscale image with values in [0,255] loaded using pillow
                Note: Use 'L' flag while converting using Pillow's function.
        """

        img = None
  

        # Open image using Pillow and convert to grayscale ('L')
        with Image.open(path) as im:
            img = im.convert("L")

  
        return img

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetches the item (image, label) at a given index.

        Note: Do not forget to apply the transforms, if they exist

        Hint:
        1) get info from self.dataset
        2) use load_img_from_path
        3) apply transforms if valid

        Args:
            index: Index
        Returns:
            img: image of shape (H,W)
            class_idxs: indices of shape (num_classes, ) of the ground truth classes for this image
        """
        img = None
        class_idxs = None

  

        path, class_idxs = self.dataset[index]

        pil_img = self.load_img_from_path(path)

        if self.transform is not None:
            img = self.transform(pil_img)
        else:
            # ensure channel-first shape (C,H,W) for grayscale images
            arr = np.asarray(pil_img)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]  # shape -> (1, H, W)
            img = torch.from_numpy(arr).float()

  
        return img, class_idxs

    def __len__(self) -> int:
        """Returns the number of items in the dataset.

        Returns:
            l: length of the dataset
        """

        l = 0

  

        l = len(self.dataset)
  
        return l
