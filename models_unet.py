import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------- Helper Blocks -----------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)


class Down(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p


class Up(nn.Module):
    """Upscaling then DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
       x1 = self.up(x1)
       x = torch.cat([x1, x2], 1)
       return self.conv(x)


    def forward(self, x1, x2):
        # x1 is the output from the previous layer (up-sampled)
        # x2 is the feature map from the corresponding encoder layer (skip connection)
        
        # 1. Up-sample x1
        x1 = self.up(x1)
        
        # 2. Match size of x1 to x2 (important if input size is not a power of 2)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # 3. Concatenate (Skip Connection)
        x = torch.cat([x2, x1], dim=1)
        
        # 4. Convolution
        return self.conv(x)


# ----------------- Main UNet Model -----------------

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = Down(in_channels, 64)
        self.down_convolution_2 = Down(64, 128)
        self.down_convolution_3 = Down(128, 256)
        self.down_convolution_4 = Down(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = Up(1024, 512)
        self.up_convolution_2 = Up(512, 256)
        self.up_convolution_3 = Up(256, 128)
        self.up_convolution_4 = Up(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
       down_1, p1 = self.down_convolution_1(x)
       down_2, p2 = self.down_convolution_2(p1)
       down_3, p3 = self.down_convolution_3(p2)
       down_4, p4 = self.down_convolution_4(p3)

       b = self.bottle_neck(p4)

       up_1 = self.up_convolution_1(b, down_4)
       up_2 = self.up_convolution_2(up_1, down_3)
       up_3 = self.up_convolution_3(up_2, down_2)
       up_4 = self.up_convolution_4(up_3, down_1)

       out = self.out(up_4)
       return out


