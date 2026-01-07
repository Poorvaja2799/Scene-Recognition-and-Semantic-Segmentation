import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        """
        super(SimpleNet, self).__init__()

        self.conv_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()
        self.loss_criterion = None

  

        self.conv_layers = nn.Sequential(
                # Conv block 1: 1x64x64 -> 10x60x60 -> 10x20x20
                nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=3),

                # Conv block 2: 10x20x20 -> 20x16x16 -> 20x5x5
                nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=3),
        )

        # 20 feature maps of size 5x5 -> 500
        self.fc_layers = nn.Sequential(
            nn.Flatten(),                 # -> (N, 500)
            nn.Linear(20 * 5 * 5, 100),  # 500 -> 100
            nn.ReLU(inplace=True),
            nn.Linear(100, 15),          # 100 -> 15 (logits)
        )

        # Use mean reduction as requested
        self.loss_criterion = nn.CrossEntropyLoss(reduction='mean')


  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass with the net

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (raw scores) of the net [Dim: (N,15)]
        """
        model_output = None
  
        
        x = self.conv_layers(x)
        model_output = self.fc_layers(x)

  

        return model_output
