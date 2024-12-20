import torch
import torch.nn as nn

from src.utils.unet_parts import DoubleConv, DownSample, UpSample


class UNet(nn.Module):
    """
    A PyTorch implementation of the U-Net architecture for image segmentation tasks.
    
    The U-Net is a fully convolutional network that combines encoding (contracting) 
    and decoding (expanding) paths with skip connections to capture both spatial 
    and contextual information.

    Attributes:
        down_convolution_1: First downsampling layer, reduces spatial dimensions and increases feature depth.
        down_convolution_2: Second downsampling layer.
        down_convolution_3: Third downsampling layer.
        down_convolution_4: Fourth downsampling layer.
        bottle_neck: Bottleneck layer with the deepest features, using double convolutions.
        up_convolution_1: First upsampling layer, expands spatial dimensions and reduces feature depth.
        up_convolution_2: Second upsampling layer.
        up_convolution_3: Third upsampling layer.
        up_convolution_4: Fourth upsampling layer.
        out: Final convolutional layer to map feature maps to the desired number of output classes.
    """
    def __init__(self, in_channels, num_classes):
        """
        Initializes the U-Net model.

        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB images).
            num_classes (int): Number of output classes for segmentation.
        """
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)
        
        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the U-Net model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, num_classes, height, width).
        """
        # Downsampling path
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        # Bottleneck
        b = self.bottle_neck(p4)

        # Upsampling path
        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        # Output layer
        out = self.out(up_4)
        return out
