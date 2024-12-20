import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    A module for two consecutive convolutional layers, each followed by a ReLU activation.

    This is a basic building block of the U-Net model, used for feature extraction 
    while preserving spatial resolution.

    Attributes:
        conv_op: A sequential container of two 2D convolutional layers with ReLU activations.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initializes the DoubleConv module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass through the DoubleConv module.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_channels, height, width).
        """
        return self.conv_op(x)


class DownSample(nn.Module):
    """
    A module for downsampling input through a convolutional operation followed by max pooling.

    This module reduces the spatial dimensions while increasing the feature depth.

    Attributes:
        conv: A DoubleConv layer for feature extraction.
        pool: A MaxPool2d layer for downsampling.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initializes the DownSample module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Forward pass through the DownSample module.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, height, width).

        Returns:
            tuple: 
                - down (torch.Tensor): Feature map after convolution (batch_size, out_channels, height, width).
                - p (torch.Tensor): Downsampled feature map (batch_size, out_channels, height//2, width//2).
        """
        down = self.conv(x)
        p = self.pool(down)

        return down, p


class UpSample(nn.Module):
    """
    A module for upsampling input using transposed convolution, followed by concatenation and convolution.

    This module increases the spatial dimensions while reducing the feature depth, combining features from 
    the encoder path via skip connections.

    Attributes:
        up: A ConvTranspose2d layer for upsampling.
        conv: A DoubleConv layer for feature refinement after concatenation.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initializes the UpSample module.

        Args:
            in_channels (int): Number of input channels (from both upsampled and skip-connected feature maps).
            out_channels (int): Number of output channels after convolution.
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Forward pass through the UpSample module.

        Args:
            x1 (torch.Tensor): Input tensor from the previous layer (batch_size, in_channels, height, width).
            x2 (torch.Tensor): Skip-connected tensor from the encoder (batch_size, in_channels//2, height*2, width*2).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_channels, height*2, width*2).
        """
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)
