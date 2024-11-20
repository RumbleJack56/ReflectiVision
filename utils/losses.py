"""  
arima losss type
loss_edge_corr = criterion_edge_corr(reduce(lambda x,y:x*y,GCLoss(*net(Input)), 0)
loss_edge_sum = criterion_edge_sum(gradB + gradR, imgs_grad_label)
loss_edge = (epoch - 1) * (loss_edge_sum + loss_edge_corr)   
"""

import torch.nn as nn
import torch

class GCLoss(nn.Module):
    def __init__(self):
        super(GCLoss, self).__init__()

        #sobelfilters , lib not workign
        sobel_x = torch.Tensor([[1,0,-1],[2,0,-2],[1,0,-1]]).view((1,1,3,3)).repeat(1,3,1,1)
        sobel_y = torch.Tensor([[1,2,1],[0,0,0],[-1,-2,-1]]).view((1,1,3,3)).repeat(1,3,1,1)

        self.gxb = nn.Conv2d(3,1,kernel_size=3,stride=1,padding=0,bias=False)
        self.gxb.weight = nn.Parameter(sobel_x)
        for param in self.gxb.parameters():
            param.requires_grad = False

        self.gyb = nn.Conv2d(3,1,kernel_size=3,stride=1,padding=0,bias=False)
        self.gyb.weight = nn.Parameter(sobel_y)
        for param in self.gyb.parameters():
            param.requires_grad = False


        self.gxr = nn.Conv2d(3,1,kernel_size=3,stride=1,padding=0,bias=False)
        self.gxr.weight = nn.Parameter(sobel_x)
        for param in self.gxr.parameters():
            param.requires_grad = False
        
        self.gyr = nn.Conv2d(3,1,kernel_size=3,stride=1,padding=0,bias=False)
        self.gyr.weight = nn.Parameter(sobel_y)
        for param in self.gyr.parameters():
            param.requires_grad = False

        self.af_B = nn.Tanhshrink()
        self.af_R = nn.Tanhshrink()
        

    def forward(self, B, R):

        gradout_B = self.af_B(self.gyb(B) + self.gxb(B))
        gradout_R = self.af_R(self.gyr(R) + self.gxr(R))
        return gradout_B, gradout_R