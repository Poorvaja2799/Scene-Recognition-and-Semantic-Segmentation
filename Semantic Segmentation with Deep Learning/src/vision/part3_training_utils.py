from typing import Tuple

import torch
from torch import nn

import src.vision.cv2_transforms as transform
from src.vision.part5_pspnet import PSPNet
from src.vision.part4_segmentation_net import SimpleSegmentationNet


def get_model_and_optimizer(args) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """
    Create your model, optimizer and configure the initial learning rates.

    Use the SGD optimizer, use a parameters list, and set the momentum and
    weight decay for each parameter group according to the parameter values
    in `args`.

    Create 5 param groups for the 0th + 1st,2nd,3rd,4th ResNet layer modules,
    and then add separate groups afterwards for the classifier and/or PPM
    heads.

    You should set the learning rate for the resnet layers to the base learning
    rate (args.base_lr), and you should set the learning rate for the new
    PSPNet PPM and classifiers to be 10 times the base learning rate.

    Args:
        args: object containing specified hyperparameters, including the "arch"
           parameter that determines whether we should return PSPNet or the
           SimpleSegmentationNet
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    
    # Create the model based on architecture specified in args
    if args.arch == "PSPNet":
        model = PSPNet(
            layers=args.layers,
            num_classes=args.classes,
            zoom_factor=args.zoom_factor,
            criterion=nn.CrossEntropyLoss(ignore_index=args.ignore_label),
            pretrained=args.pretrained,
            use_ppm=args.use_ppm
        )
    elif args.arch == "SimpleSegmentationNet":
        model = SimpleSegmentationNet(
            pretrained=args.pretrained,
            num_classes=args.classes,
            criterion=nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        )
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")
    
    # Create parameter groups for different learning rates
    param_groups = []
    
    # ResNet layers (layer0, layer1, layer2, layer3, layer4) - base learning rate
    if hasattr(model, 'layer0'):
        param_groups.append({
            'params': model.layer0.parameters(),
            'lr': args.base_lr,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay
        })
    
    # For PSPNet, we access ResNet layers through model.layer1, etc.
    # For SimpleSegmentationNet, we access through model.resnet.layer1, etc.
    if args.arch == "PSPNet":
        resnet_layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    else:  # SimpleSegmentationNet
        resnet_layers = [model.resnet.layer1, model.resnet.layer2, model.resnet.layer3, model.resnet.layer4]
    
    for layer in resnet_layers:
        param_groups.append({
            'params': layer.parameters(),
            'lr': args.base_lr,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay
        })
    
    # Classifier and PPM modules - 10x base learning rate
    classifier_params = []
    
    # Add classifier parameters
    if hasattr(model, 'cls') and model.cls is not None:
        classifier_params.extend(list(model.cls.parameters()))
    
    # Add PPM parameters (only for PSPNet)
    if hasattr(model, 'ppm') and model.ppm is not None:
        classifier_params.extend(list(model.ppm.parameters()))
    
    # Add auxiliary classifier parameters (only for PSPNet)
    if hasattr(model, 'aux') and model.aux is not None:
        classifier_params.extend(list(model.aux.parameters()))
    
    if classifier_params:
        param_groups.append({
            'params': classifier_params,
            'lr': args.base_lr * 10,  # 10x base learning rate
            'momentum': args.momentum,
            'weight_decay': args.weight_decay
        })
    
    # Create SGD optimizer with parameter groups
    optimizer = torch.optim.SGD(param_groups)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return model, optimizer


def update_learning_rate(current_lr: float, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
    """
    Given an updated current learning rate, set the ResNet modules to this
    current learning rate, and the classifiers/PPM module to 10x the current
    lr.

    Hint: You can loop over the dictionaries in the optimizer.param_groups
    list, and set a new "lr" entry for each one. They will be in the same order
    you added them above, so if the first N modules should have low learning
    rate, and the next M modules should have a higher learning rate, this
    should be easy modify in two loops.

    Note: this depends upon how you implemented the param groups above.
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    
    # The parameter groups are in the order we added them:
    # First 5 groups: layer0, layer1, layer2, layer3, layer4 (ResNet layers)
    # Last group: classifier/PPM modules
    
    # Update ResNet layers (first 5 parameter groups) to current_lr
    for i in range(min(5, len(optimizer.param_groups) - 1)):
        optimizer.param_groups[i]['lr'] = current_lr
    
    # Update classifier/PPM modules (last parameter group) to 10x current_lr
    if len(optimizer.param_groups) > 5:
        optimizer.param_groups[-1]['lr'] = current_lr * 10


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return optimizer


def get_train_transform(args) -> transform.Compose:
    """
    Compose together with transform.Compose() a series of data proprocessing
    transformations for the training split, with data augmentation. Use the classes
    from `src/vision/cv2_transforms`, imported above as `transform`.

    These should include resizing the short side of the image to args.short_size,
    then random horizontal flipping, blurring, rotation, scaling (in any order),
    followed by taking a random crop of size (args.train_h, args.train_w), converting
    the Numpy array to a Pytorch tensor, and then normalizing by the
    Imagenet mean and std (provided here).

    Note that your scaling should be confined to the [scale_min,scale_max] params in the
    args. Also, your rotation should be confined to the [rotate_min,rotate_max] params.

    To prevent black artifacts after a rotation or a random crop, specify the paddings
    to be equal to the Imagenet mean to pad any black regions.

    You should set such artifact regions of the ground truth to be ignored.

    Use the classes
    from `src/vision/cv2_transforms`, imported above as `transform`.

    Args:
        args: object containing specified hyperparameters

    Returns:
        train_transform
    """

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    
    # Create training transforms with data augmentation
    train_transform = transform.Compose([
        # 1. Resize the short side to args.short_size
        transform.ResizeShort(args.short_size),
        
        # 2. Data augmentation transforms (order can vary)
        transform.RandomHorizontalFlip(p=0.5),
        transform.RandomGaussianBlur(radius=5),
        transform.RandRotate(
            rotate=(args.rotate_min, args.rotate_max),
            padding=mean,  # Use ImageNet mean for padding
            ignore_label=args.ignore_label,
            p=0.5
        ),
        transform.RandScale(scale=(args.scale_min, args.scale_max)),
        
        # 3. Random crop to target size
        transform.Crop(
            size=(args.train_h, args.train_w),
            crop_type="rand",
            padding=mean,  # Use ImageNet mean for padding
            ignore_label=args.ignore_label
        ),
        
        # 4. Convert to tensor
        transform.ToTensor(),
        
        # 5. Normalize with ImageNet mean and std
        transform.Normalize(mean=mean, std=std)
    ])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return train_transform


def get_val_transform(args) -> transform.Compose:
    """
    Compose together with transform.Compose() a series of data proprocessing
    transformations for the val split, with no data augmentation. Use the classes
    from `src/vision/cv2_transforms`, imported above as `transform`.

    These should include resizing the short side of the image to args.short_size,
    taking a *center* crop of size (args.train_h, args.train_w) with a padding equal
    to the Imagenet mean, converting the Numpy array to a Pytorch tensor, and then
    normalizing by the Imagenet mean and std (provided here).

    Args:
        args: object containing specified hyperparameters

    Returns:
        val_transform
    """

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    
    # Create validation transforms without data augmentation
    val_transform = transform.Compose([
        # 1. Resize the short side to args.short_size
        transform.ResizeShort(args.short_size),
        
        # 2. Center crop to target size (no randomness)
        transform.Crop(
            size=(args.train_h, args.train_w),
            crop_type="center",
            padding=mean,  # Use ImageNet mean for padding
            ignore_label=args.ignore_label
        ),
        
        # 3. Convert to tensor
        transform.ToTensor(),
        
        # 4. Normalize with ImageNet mean and std
        transform.Normalize(mean=mean, std=std)
    ])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return val_transform
