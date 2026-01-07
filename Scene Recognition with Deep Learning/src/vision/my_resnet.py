import torch
import torch.nn as nn
from torchvision.models import resnet18


class MyResNet18(nn.Module):
    def __init__(self):
        """Initialize network layers.

        Note: Do not forget to freeze the layers of ResNet except the last one
        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        Download pretrained resnet using pytorch's API (Hint: see the import statements)
        """
        super().__init__()

        self.conv_layers = None
        self.fc_layers = None
        self.loss_criterion = None

  

        # Load pretrained ResNet18
        resnet = resnet18(pretrained=True)

        # Freeze all parameters initially
        for param in resnet.parameters():
            param.requires_grad = False

        # We'll keep the pretrained resnet features (all layers except final fc)
        # conv_layers: everything up to avgpool
        self.conv_layers = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )

        # Replace the final fully-connected layer to match 15 outputs
        in_features = resnet.fc.in_features
        self.fc_layers = nn.Linear(in_features, 15)

        for p in self.conv_layers.parameters():
            p.requires_grad = False

        # Only the new fc layer should require grad
        for param in self.fc_layers.parameters():
            param.requires_grad = True

        # Loss criterion: CrossEntropy with mean reduction
        self.loss_criterion = nn.CrossEntropyLoss(reduction="mean")

  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass with the net, duplicating grayscale channel to 3-channel.

        Args:
            x: tensor of shape (N,C,H,W) representing input batch of images
        Returns:
            y: tensor of shape (N,num_classes) representing the output (raw scores) of the net
                Note: we set num_classes=15
        """
        model_output = None
        x = x.repeat(1, 3, 1, 1)  # as ResNet accepts 3-channel color images
  

        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        model_output = self.fc_layers(x)

  
        return model_output
