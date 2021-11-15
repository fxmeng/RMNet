import torch
import torch.nn as nn
from torch.nn import functional as F

class RMBlockFromScratch(nn.Module):
    def __init__(self, in_planes, mid_planes, out_planes, stride=1):
        super(RMBlockFromScratch, self).__init__()

        assert mid_planes > in_planes

        self.in_planes = in_planes
        self.mid_planes = mid_planes - out_planes +in_planes
        self.out_planes = out_planes
        self.stride = stride

        self.conv1 = nn.Conv2d(in_planes, self.mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.mid_planes)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(self.mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        nn.init.dirac_(self.conv1.weight.data[:self.in_planes])
        x = self.conv1(x)
        if self.training:
            x_mean = x.mean(dim=(0,2,3))
            x_var = x.var(dim=(0,2,3),unbiased=False)
            self.bn1.weight.data[:self.in_planes]=torch.sqrt(x_var[:self.in_planes]+self.bn1.eps)
            self.bn1.bias.data[:self.in_planes]=x_mean[:self.in_planes]
        else:
            self.bn1.weight.data[:self.in_planes]=torch.sqrt(self.bn1.running_var[:self.in_planes]+self.bn1.eps)
            self.bn1.bias.data[:self.in_planes]=self.bn1.running_mean[:self.in_planes]
        x = F.batch_norm(x,self.bn1.running_mean,self.bn1.running_var,self.bn1.weight,self.bn1.bias,training=self.training)
        out = self.relu1(x)
        if self.in_planes==self.out_planes and self.stride==1:
            nn.init.dirac_(self.conv2.weight.data[:,:self.in_planes])
        out = self.bn2(self.conv2(out))
        return self.relu2(out)
