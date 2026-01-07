import torch
import torch.nn as nn
from torchvision.models import resnet18


class MultilabelResNet18(nn.Module):
    def __init__(self):
        """Initialize network layers.

        Note: Do not forget to freeze the layers of ResNet except the last one
        Note: Consider which activation function to use
        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        Download pretrained resnet using pytorch's API (Hint: see the import statements)
        """
        super().__init__()

        self.conv_layers = None
        self.fc_layers = None
        self.loss_criterion = None
        self.activation = None

  

        backbone = resnet18(pretrained=True)

        # conv feature extractor up to avgpool
        self.conv_layers = nn.Sequential(*list(backbone.children())[:-1])  # -> (N, 512, 1, 1)

        # new multilabel head: 7 attributes
        in_feats = backbone.fc.in_features  # 512
        self.fc_layers = nn.Linear(in_feats, 7)

        # freeze backbone; train only the new head
        for p in self.conv_layers.parameters():
            p.requires_grad = False
        for p in self.fc_layers.parameters():
            p.requires_grad = True

        # BCE with logits (mean reduction)
        # self.loss_criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.loss_criterion = nn.BCELoss(reduction="mean")

  

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
  
        
        # duplicate grayscale to 3 channels if needed
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        feats = self.conv_layers(x)        # (N, 512, 1, 1)
        feats = feats.flatten(1)           # (N, 512)
        model_output = torch.sigmoid(self.fc_layers(feats))     # (N, 7) raw scores
        # model_output = self.fc_layers(feats)

  
        return model_output
