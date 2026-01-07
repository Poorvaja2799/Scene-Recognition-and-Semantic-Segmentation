import logging
import os
import pdb
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

from src.vision.part5_pspnet import PSPNet
from src.vision.utils import load_class_names, get_imagenet_mean_std, get_logger, normalize_img


_ROOT = Path(__file__).resolve().parent.parent.parent

logger = get_logger()


def load_pretrained_model(args, device: torch.device):
    """Load Pytorch pre-trained PSPNet model from disk of type torch.nn.DataParallel.

    Note that `args.num_model_classes` will be size of logits output.

    Args:
        args:
        device:

    Returns:
        model
    """
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model = PSPNet(
        layers=args.layers,
        num_classes=args.classes,
        zoom_factor=args.zoom_factor,
        criterion=criterion,
        pretrained=False
    )

    # logger.info(model)
    if device.type == 'cuda':
        cudnn.benchmark = True

    if os.path.isfile(args.model_path):
        logger.info(f"=> loading checkpoint '{args.model_path}'")
        checkpoint = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        logger.info(f"=> loaded checkpoint '{args.model_path}'")
    else:
        raise RuntimeError(f"=> no checkpoint found at '{args.model_path}'")
    model = model.to(device)

    return model



def model_and_optimizer(args, model) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """
    This function is similar to get_model_and_optimizer in Part 3.

    Use the model trained on Camvid as the pretrained PSPNet model, change the
    output classes number to 2 (the number of classes for Kitti).
    Refer to Part 3 for optimizer initialization.

    Args:
        args: object containing specified hyperparameters
        model: pre-trained model on Camvid

    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    
    # Modify the model for transfer learning
    # Change the number of output classes from CamVid (11 classes) to KITTI (2 classes)
    
    # Replace the main classifier (cls) to output 2 classes (for KITTI)
    in_feats = model.cls[0].in_channels  
    model.cls = model._PSPNet__create_classifier(in_feats=in_feats, out_feats=512, num_classes=2)
    
    # Replace the auxiliary classifier (aux) to output 2 classes (for KITTI)
    aux_in_feats = model.aux[0].in_channels
    model.aux = model._PSPNet__create_classifier(in_feats=aux_in_feats, out_feats=256, num_classes=2)
    
    # Update the criterion to match the new number of classes
    model.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    
    # Create parameter groups for different learning rates (similar to Part 3)
    param_groups = []
    
    # ResNet layers (layer0, layer1, layer2, layer3, layer4) - base learning rate
    if hasattr(model, 'layer0'):
        param_groups.append({
            'params': model.layer0.parameters(),
            'lr': args.base_lr,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay
        })
    
    # Add other ResNet layers
    resnet_layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    for layer in resnet_layers:
        param_groups.append({
            'params': layer.parameters(),
            'lr': args.base_lr,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay
        })
    
    # Classifier and PPM modules - 10x base learning rate
    classifier_params = []
    
    # Add classifier parameters (newly initialized, so higher learning rate)
    if hasattr(model, 'cls') and model.cls is not None:
        classifier_params.extend(list(model.cls.parameters()))
    
    # Add PPM parameters 
    if hasattr(model, 'ppm') and model.ppm is not None:
        classifier_params.extend(list(model.ppm.parameters()))
    
    # Add auxiliary classifier parameters (newly initialized)
    if hasattr(model, 'aux') and model.aux is not None:
        classifier_params.extend(list(model.aux.parameters()))
    
    if classifier_params:
        param_groups.append({
            'params': classifier_params,
            'lr': args.base_lr * 10,  # 10x base learning rate for new/modified layers
            'momentum': args.momentum,
            'weight_decay': args.weight_decay
        })
    
    # Create SGD optimizer with parameter groups
    optimizer = torch.optim.SGD(param_groups)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return model, optimizer
