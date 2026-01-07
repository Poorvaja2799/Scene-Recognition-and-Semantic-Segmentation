import torch
import torch.nn as nn


class SimpleNetFinal(nn.Module):
    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means
        """
        super(SimpleNetFinal, self).__init__()

        self.conv_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()
        self.loss_criterion = None

  

        self.conv_layers = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0),
                nn.BatchNorm2d(10),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Dropout(p=0.5),

                nn.Conv2d(in_channels=10, out_channels=15, kernel_size=5, stride=1, padding=0),
                nn.BatchNorm2d(15),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),

                nn.Conv2d(in_channels=15, out_channels=20, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(20 * 5 * 5, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 15),
        )

        # Use mean reduction as requested
        self.loss_criterion = nn.CrossEntropyLoss(reduction='mean')

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
  

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
