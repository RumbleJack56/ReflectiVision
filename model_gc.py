import torch.nn as nn
import torch
from torchvision.models import vgg16
from utils.utils import Interpolate

class ConvBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, activation=nn.LeakyReLU(negative_slope=0.01,inplace=True)):
        super(ConvBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1),
            nn.BatchNorm2d(middle_channels),
            activation,
            nn.Conv2d(middle_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            activation)
    def forward(self, x): return self.model(x)


class GCNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(GCNet, self).__init__()


        filt = [32, 64, 128, 256, 512]   #  filtersssssss

        self.pool = nn.MaxPool2d(2, 2)
        #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = Interpolate(scale_factor=2, mode='bilinear')


        # self.conv0_0 = ConvBlock(in_channels, filt[0], filt[0])
        # self.conv1_0 = ConvBlocks(filt[0], filt[1], filt[1])
        # self.conv2_0 = ConvBlock(filt[1], filt[2], filt[2])
        # self.conv3_0 = ConvBlock(filt[2], filt[3], filt[3])
        # self.conv4_0 = ConvBlock(filt[3], filt[4], filt[4])
        
        
        self.conv0_0 = ConvBlock(in_channels, filt[0], filt[0])
        self.conv1_0= ConvBlock(filt[0]  , filt[1], filt[1])
        self.conv2_0= ConvBlock(filt[1],  filt[2], filt[2])
        self.conv3_0 = ConvBlock(filt[2] , filt[3], filt[3])
        self.conv4_0= ConvBlock( filt[3] , filt[4], filt[4])

        self.conv0_1 = ConvBlock(filt[0]+filt[1], filt[0], filt[0])
        self.conv1_1 = ConvBlock(filt[1]+filt[2], filt[1], filt[1])
        # self.conv1_1 = ConvBlock(filt[1]+filt[2], filt[3], filt[3])
        self.conv2_1 = ConvBlock(filt[2]+filt[3], filt[2], filt[2])
        # self.conv2_1 = ConvBlock(filt[2]+filt[3], filt[3], filt[3])
        self.conv3_1 = ConvBlock(filt[3]+filt[4], filt[3], filt[3])

        self.conv0_2 = ConvBlock(filt[0]*2+filt[1], filt[0], filt[0])
        self.conv1_2 = ConvBlock(filt[1]*2+filt[2], filt[1], filt[1])
        self.conv2_2 = ConvBlock(filt[2]*2+filt[3], filt[2], filt[2])
        # self.conv2_2 = ConvBlock(filt[2]*2+filt[3], filt[3], filt[3])


        self.conv0_3 = ConvBlock(filt[0]*3+filt[1], filt[0], filt[0])
        self.conv1_3 = ConvBlock(filt[1]*3+filt[2], filt[1], filt[1])

        self.conv0_4 = ConvBlock(filt[0]*4+filt[1], filt[0], filt[0])

        self.final1 = nn.Sequential(nn.Conv2d(filt[0], out_channels, kernel_size=3, padding=1))
        self.final2 = nn.Sequential(nn.Conv2d(filt[0], out_channels, kernel_size=3, padding=1))
        self.final3 = nn.Sequential(nn.Conv2d(filt[0], out_channels, kernel_size=3, padding=1))
        self.final4 = nn.Sequential(nn.Conv2d(filt[0], filt[0], 5, padding=2),
            nn.BatchNorm2d(filt[0]),
            nn.LeakyReLU(negative_slope=0.01,inplace=True),
            nn.Conv2d(filt[0], out_channels, kernel_size=3, padding=1))

        self.G_x_D = nn.Conv2d(3,1,kernel_size=3,stride=1,padding=0,bias=False)
        self.G_y_D = nn.Conv2d(3,1,kernel_size=3,stride=1,padding=0,bias=False)
        self.G_x_G = nn.Conv2d(3,1,kernel_size=3,stride=1,padding=0,bias=False)
        self.G_y_G = nn.Conv2d(3,1,kernel_size=3,stride=1,padding=0,bias=False)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output4 = self.final4(x0_4)

        return output4



class ConvBlockVG(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, activation=nn.LeakyReLU(negative_slope=0.01, inplace=True)):
        super(ConvBlockVG, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1),
            nn.BatchNorm2d(middle_channels),
            activation,
            nn.Conv2d(middle_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            activation
        )

    def forward(self, x): return self.model(x)


class GCNet_vgg16(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(GCNet_vgg16, self).__init__()

        # wrong here -> fix: use upsample with scale factor instead of maxpool
        vgg16 = vgg16(pretrained=True).features
        self.vgg_backbone = nn.Sequential(*list(vgg16.children())[:16]) #16 init layers

        filt = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        # initial convolution layers
        self.conv0_1 = ConvBlockVG(filt[0] + 64, filt[0], filt[0])
        # self.conv1_1 = ConvBlockVG(filt[0] + filt[1], filt[1], filt[1])
        self.conv1_1 = ConvBlockVG(filt[0] + filt[1], filt[0], filt[0])  # fixed

        self.conv2_1 = ConvBlockVG(filt[1] + filt[2], filt[2], filt[2])
        self.conv3_1 = ConvBlockVG(filt[2] + filt[3], filt[3], filt[3])
        
        
        self.conv0_2 = ConvBlockVG(filt[0] * 2 + filt[1], filt[0], filt[0])
        # self.conv0_2 = ConvBlockVG(filt[0] * 2 + filt[2], filt[1], filt[1])

        self.conv1_2 = ConvBlockVG(filt[1] * 2 + filt[2], filt[1], filt[1])
        self.conv2_2 = ConvBlockVG(filt[2] * 2 + filt[3], filt[2], filt[2])

        self.conv0_3 = ConvBlockVG(filt[0] * 3 + filt[1], filt[0], filt[0])

        self.conv1_3 = ConvBlockVG(filt[1] * 3 + filt[2], filt[1], filt[1])

        self.conv0_4 = ConvBlockVG(filt[0] * 4 + filt[1], filt[0], filt[0])

        # more than 4 layers, removing extra output
        self.final4 = nn.Sequential(
            nn.Conv2d(filt[0], out_channels, kernel_size=3, padding=1)
        )

        # extra layers for feature mapping
        # this needs a different kernel size
        self.G_x_D = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.G_y_D = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.G_x_G = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.G_y_G = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=0, bias=False)

    def forward(self, x):
        # pass through vgg16 for feature extraction
        x0_0 = self.vgg_backbone(x)
        x1_0 = self.conv0_1(torch.cat([x0_0, self.up(x0_0)], 1))

        # apply convolution blocks
        x2_0 = self.conv1_1(torch.cat([x1_0, self.up(x1_0)], 1)) # changed weights
        x3_0 = self.conv2_1(torch.cat([x2_0, self.up(x2_0)], 1))
        x4_0 = self.conv3_1(torch.cat([x3_0, self.up(x3_0)], 1))
        x0_1 = self.conv0_2(torch.cat([x0_0, x1_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_2(torch.cat([x1_0, x2_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_2(torch.cat([x2_0, x3_0, self.up(x3_0)], 1))

        # more skip connections
        x0_2 = self.conv0_3(torch.cat([x0_0, x1_0, x2_0, self.up(x2_0)], 1))
        x1_2 = self.conv1_3(torch.cat([x1_0, x2_0, x3_0, self.up(x3_0)], 1))
        x0_3 = self.conv0_4(torch.cat([x0_0, x1_0, x2_0, x3_0, self.up(x4_0)], 1))
        output = self.final4(x0_3)
        return self.final4(x0_3)