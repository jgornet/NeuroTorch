from torch import nn
from torch.nn import functional as F
from torch.nn import Conv3d, ConvTranspose3d, MaxPool3d, Upsample

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels):
        kernel_size = (3, 3, 3)
        padding = kernel_size[0] - 1
        self.convolution = Conv3d(in_channels, out_channels, kernel_size,
                                  padding)
        self.max_pool = MaxPool3d(kernel_size=(2, 2))

    def forward(self, x):
        x1 = self.convolution(x)
        x1 = self
