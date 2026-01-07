import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image


def compute_mean_and_std(dir_name: str) -> Tuple[float, float]:
    """Compute the mean and the standard deviation of all images present within the directory.

    Note: convert the image in grayscale and then in [0,1] before computing mean
    and standard deviation

    Mean = (1/n) * \sum_{i=1}^n x_i
    Variance = (1 / (n-1)) * \sum_{i=1}^n (x_i - mean)^2
    Standard Deviation = sqrt( Variance )

    Args:
        dir_name: the path of the root dir

    Returns:
        mean: mean value of the gray color channel for the dataset
        std: standard deviation of the gray color channel for the dataset
    """
    mean = None
    std = None

    # Gather all image file paths under the directory (search recursively)
    patterns = [os.path.join(dir_name, "**", "*.jpg"), os.path.join(dir_name, "**", "*.jpeg"), os.path.join(dir_name, "**", "*.png")]
    file_list = []
    for p in patterns:
        file_list.extend(glob.glob(p, recursive=True))

    # If no images found, return zeros
    if len(file_list) == 0:
        return 0.0, 0.0

    # Accumulate sum and sum of squares over all pixels across images
    total_pixels = 0
    sum_pixels = 0.0
    sumsq_pixels = 0.0

    for fp in file_list:
        try:
            img = Image.open(fp).convert("L")
        except Exception:
            # skip unreadable files
            continue
        arr = np.asarray(img, dtype=np.float64) / 255.0  # scale to [0,1]
        pixels = arr.size
        total_pixels += pixels
        sum_pixels += arr.sum()
        sumsq_pixels += (arr ** 2).sum()

    if total_pixels == 0:
        return 0.0, 0.0

    mean = sum_pixels / total_pixels

    # sample variance (using n-1) per problem statement
    # variance = (1/(n-1)) * (sum(x^2) - n * mean^2)
    if total_pixels > 1:
        variance = (sumsq_pixels - total_pixels * (mean ** 2)) / (total_pixels - 1)
        # numerical safety
        variance = max(variance, 0.0)
        std = float(np.sqrt(variance))
    else:
        std = 0.0
    return mean, std
