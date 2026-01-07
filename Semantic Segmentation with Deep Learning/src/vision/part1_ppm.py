from typing import List

import torch
from torch import nn
import torch.nn.functional as F


class PPM(nn.Module):
    """
    A hierarchical global prior, containing information with different scale
    and varying among different sub-regions.

    The pyramid pooling module fuses features under four different pyramid
    scales.

    The coarsest level highlighted in red is global pooling to generate a
    single bin output. The following pyramid level separates the feature map
    into different sub-regions and forms pooled representation for different
    locations. The output of different levels in the pyramid pooling module
    contains the feature map with varied sizes. To maintain the weight of
    global feature, we use 1×1 convolution layer after each pyramid level to
    reduce the dimension of context representation to 1/N of the original one
    if the level size of pyramid is N.
    """

    def __init__(self, in_dim: int, reduction_dim: int, bins: List[int]) -> None:
        """
        If bins=(1, 2, 3, 6), then the PPM will operate at four levels, with
        bin sizes of 1×1, 2×2, 3×3 and 6×6, respectively.

        The PPM utilizes nn.AdaptiveAvgPool2d(bin) to break an image into
        (bin x bin) subregions, and then pools all entries inside each
        subregion.

        For each bin size, you should create a sequential module of
        (AdaptiveAvgPool2d, 2d conv w/ 1x1 kernel, 2d batch norm, and ReLU).
        These modules should be stored inside the self.features attribute.
        """
        super(PPM, self).__init__()
        self.features = []

        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################
        
        # Create a sequential module for each bin size
        for bin_size in bins:
            # For each bin size, create: AdaptiveAvgPool2d -> Conv2d -> BatchNorm2d -> ReLU
            module = nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_size),  # Pool into bin_size x bin_size subregions
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),  # 1x1 conv to reduce dimensions
                nn.BatchNorm2d(reduction_dim),  # Batch normalization
                nn.ReLU(inplace=True)  # ReLU activation
            )
            self.features.append(module)

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################
        self.features = nn.ModuleList(self.features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the forward pass of the PPM module.

        Feed the input through each module in the module list, upsample to the
        desired output size, and append each output to a list of outputs.
        Finally, concatenate them along the channel dimension. The first item
        in the output list should be the input itself.

        For upsampling, use Pytorch's bilinear interpolation, and ensure that
        corners are aligned.

        Args:
            x: tensor of shape (N,in_dim,H,W)

        Returns:
            out: tensor of shape (N,C,H,W) where
                C = in_dim + len(bins)*reduction_dim
        """

        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################
        
        # Get input dimensions
        N, C, H, W = x.shape
        
        # Start output list with the original input
        outputs = [x]
        
        # Apply each pooling module and upsample results
        for module in self.features:
            # Apply the pooling module (AdaptiveAvgPool2d -> Conv2d -> BatchNorm2d -> ReLU)
            pooled = module(x)
            
            # Upsample back to original spatial dimensions using bilinear interpolation
            # align_corners=False ensures corners are aligned as specified
            upsampled = F.interpolate(
                pooled, 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            )
            
            outputs.append(upsampled)
        
        # Concatenate all outputs along the channel dimension
        output = torch.cat(outputs, dim=1)

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################
        return output
