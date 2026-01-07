"""
Contains functions with different data transforms
"""

from typing import Sequence, Tuple

import numpy as np
import torchvision.transforms as transforms


def get_fundamental_transforms(inp_size: Tuple[int, int]) -> transforms.Compose:
    """Returns the core transforms necessary to feed the images to our model.
    Args:
        inp_size: tuple denoting the dimensions for input to the model

    Returns:
        fundamental_transforms: transforms.compose with the fundamental transforms
    """
    fundamental_transforms = None
  

    # Resize to requested input size and convert PIL Image to tensor.
    fundamental_transforms = transforms.Compose(
        [
            transforms.Resize(inp_size), 
            transforms.ToTensor()
        ]
    )

  
    return fundamental_transforms


def get_fundamental_augmentation_transforms(
    inp_size: Tuple[int, int]
) -> transforms.Compose:
    """Returns the data augmentation + core transforms needed to be applied on the train set.
    Suggestions: Jittering, Flipping, Cropping, Rotating.
    Args:
        inp_size: tuple denoting the dimensions for input to the model

    Returns:
        aug_transforms: transforms.compose with all the transforms
    """
    fund_aug_transforms = None
  

    # Basic augmentation pipeline: random horizontal flip, small rotation,
    # then resize and convert to tensor.
    fund_aug_transforms = transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize(inp_size),
            transforms.ToTensor(),
        ]
    )

  
    return fund_aug_transforms


def get_fundamental_normalization_transforms(
    inp_size: Tuple[int, int], pixel_mean: Sequence[float], pixel_std: Sequence[float]
) -> transforms.Compose:
    """Returns the core transforms necessary to feed the images to our model alomg with
    normalization.

    Args:
        inp_size: tuple denoting the dimensions for input to the model
        pixel_mean: image channel means, over all images of the raw dataset
        pixel_std: image channel standard deviations, for all images of the raw dataset

    Returns:
        fundamental_transforms: transforms.compose with the fundamental transforms
    """
    fund_norm_transforms = None
  

    # Resize, convert to tensor and normalize using provided mean/std.
    fund_norm_transforms = transforms.Compose(
        [
            transforms.Resize(inp_size), 
            transforms.ToTensor(), 
            transforms.Normalize(pixel_mean, pixel_std)
        ]
    )

  
    return fund_norm_transforms


def get_all_transforms(
    inp_size: Tuple[int, int], pixel_mean: Sequence[float], pixel_std: Sequence[float]
) -> transforms.Compose:
    """Returns the data augmentation + core transforms needed to be applied on the train set,
    along with normalization. This should just be your previous method + normalization.
    Suggestions: Jittering, Flipping, Cropping, Rotating.
    Args:
        inp_size: tuple denoting the dimensions for input to the model
        pixel_mean: image channel means, over all images of the raw dataset
        pixel_std: image channel standard deviations, for all images of the raw dataset

    Returns:
        aug_transforms: transforms.compose with all the transforms
    """
    all_transforms = None
  

    # Combine augmentation pipeline with normalization at the end.
    all_transforms = transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize(inp_size),
            transforms.ToTensor(),
            transforms.Normalize(pixel_mean, pixel_std),
        ]
    )

  
    return all_transforms
