import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class SEBlock(nn.Module):
    def __init__(self, input_channels, internal_neurons,ratio):
        super(SEBlock, self).__init__()
        self.input_channels = input_channels
        self.internal_neurons=internal_neurons
        self.rmplanes=int(input_channels*ratio)
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels-self.rmplanes, kernel_size=1, stride=1, bias=True)
        
    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels-self.rmplanes, 1, 1)
        return torch.cat([inputs[:,:self.rmplanes],inputs[:,self.rmplanes:] * x],dim=1)
    
    def deploy(self):
        up = nn.Conv2d(in_channels=self.internal_neurons, out_channels=self.input_channels, kernel_size=1, stride=1, bias=True)
        nn.init.zeros_(up.weight.data[:self.rmplanes])
        up.weight.data[self.rmplanes:]=self.up.weight.data
        nn.init.constant_(up.bias.data[:self.rmplanes],100)
        up.bias.data[self.rmplanes:]=self.up.bias.data
        self.rmplanes=0
        self.up=up
        
class RMBlock(nn.Module):
    def __init__(self, in_planes, out_planes, ratio=0.5, stride=1, dilation=1, use_se=False):

        super(RMBlock, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.dilation = dilation
        self.use_se=use_se
        self.rmplanes=int(out_planes*ratio)
        assert not ratio or (in_planes==out_planes and stride==1)
        
        self.conv33 = nn.Conv2d(in_planes, self.out_planes-self.rmplanes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn33 = nn.BatchNorm2d(self.out_planes-self.rmplanes)
        self.conv11 = nn.Conv2d(in_planes, self.out_planes-self.rmplanes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn11 = nn.BatchNorm2d(self.out_planes-self.rmplanes)
        if self.in_planes==self.out_planes and stride==1:
            self.bn00 = nn.BatchNorm2d(self.out_planes-self.rmplanes)
        self.se = SEBlock(out_planes,out_planes//16,ratio) if use_se else nn.Sequential()
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.bn33(self.conv33(x))
        out += self.bn11(self.conv11(x))
        if self.in_planes==self.out_planes and self.stride==1:
            out += self.bn00(x[:,self.rmplanes:])
            out = torch.cat([x[:,:self.rmplanes], out],dim=1)
        return self.relu(self.se(out))
    
    def res2rep(self):
        if self.rmplanes:
            conv33 = nn.Conv2d(self.in_planes, self.out_planes, kernel_size=3, stride=self.stride, padding=self.dilation, dilation=self.dilation, bias=False)
            bn33 = nn.BatchNorm2d(self.out_planes)
            conv11 = nn.Conv2d(self.in_planes, self.out_planes, kernel_size=1, stride=self.stride, padding=0, bias=False)
            bn11 = nn.BatchNorm2d(self.out_planes)
            bn00 = nn.BatchNorm2d(self.out_planes)
            
            nn.init.zeros_(conv33.weight.data)
            conv33.weight.data[self.rmplanes:]=self.conv33.weight.data
            bn33.weight.data[self.rmplanes:]=self.bn33.weight.data
            bn33.bias.data[self.rmplanes:]=self.bn33.bias.data
            bn33.running_mean.data[self.rmplanes:]=self.bn33.running_mean.data
            bn33.running_var.data[self.rmplanes:]=self.bn33.running_var.data
            
            nn.init.zeros_(conv11.weight.data)
            conv11.weight.data[self.rmplanes:]=self.conv11.weight.data
            bn11.weight.data[self.rmplanes:]=self.bn11.weight.data
            bn11.bias.data[self.rmplanes:]=self.bn11.bias.data
            bn11.running_mean.data[self.rmplanes:]=self.bn11.running_mean.data
            bn11.running_var.data[self.rmplanes:]=self.bn11.running_var.data
            
            
            bn00.weight.data[self.rmplanes:]=self.bn00.weight.data
            bn00.bias.data[self.rmplanes:]=self.bn00.bias.data
            bn00.running_mean.data[self.rmplanes:]=self.bn00.running_mean.data
            bn00.running_var.data[self.rmplanes:]=self.bn00.running_var.data
            
            self.conv33=conv33
            self.bn33=bn33
            self.conv11=conv11
            self.bn11=bn11
            self.bn00=bn00
            if self.use_se:
                self.se.deploy()
            self.rmplanes=0
        return self
    
    def rep2vgg(self):
        if self.rmplanes:
            self.res2rep()
        self.eval()
        
        conv33_bn33 = torch.nn.utils.fuse_conv_bn_eval(self.conv33, self.bn33).eval()
        conv11_bn11 = torch.nn.utils.fuse_conv_bn_eval(self.conv11, self.bn11).eval()
        conv33_bn33.weight.data += F.pad(conv11_bn11.weight.data, [1, 1, 1, 1])
        conv33_bn33.bias.data += conv11_bn11.bias.data
        if self.in_planes == self.out_planes and self.stride == 1:
            conv00 = nn.Conv2d(self.in_planes, self.out_planes, kernel_size=3, padding=1, dilation=self.dilation, bias=False).eval()
            nn.init.dirac_(conv00.weight.data)
            conv00_bn00 = torch.nn.utils.fuse_conv_bn_eval(conv00, self.bn00)
            conv33_bn33.weight.data += conv00_bn00.weight.data
            conv33_bn33.bias.data += conv00_bn00.bias.data
        if self.use_se:
            return [conv33_bn33,self.se,nn.ReLU(True)]
        else:
            return [conv33_bn33,nn.ReLU(True)]
        
class RMRepSE(nn.Module):
    def __init__(self, num_blocks, num_classes=1000, base_wide=64,use_se=False):
        super(RMRepSE, self).__init__()
        feature=[]
        in_planes=3
        for t,stride,dilation,ratio in num_blocks:
            out_planes=int(t*base_wide)
            feature.append(RMBlock(in_planes, out_planes, ratio, stride, dilation, use_se))
            in_planes=out_planes
                
        self.feature=nn.Sequential(*feature)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.flat = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(out_planes, num_classes)

    def forward(self, x):
        out = self.feature(x)
        out = self.gap(out)
        out = self.flat(out)
        out = self.fc(out)
        return out
    
    def res2rep(self):
        for m in self.feature:
            if isinstance(m,RMBlock):
                m.res2rep()
                
    def deploy(self):
        blocks=[]
        for m in self.feature:
            if isinstance(m,RMBlock):
                blocks+=m.rep2vgg()
        blocks.append(self.gap)
        blocks.append(self.flat)
        blocks.append(self.fc)
        return nn.Sequential(*blocks)

def rmrepse(num_classes=1000, ratio=0, use_se=False):
    return RMRepSE([[1,2,1,0]]*1+
                 [[2.5,2,1,0],[2.5,1,1,0]]*1+
                 [[2.5,1,1,ratio],[2.5,1,1,0]]*1+
                 [[5,  2,1,0],[5,  1,1,0]]*1+
                 [[5,  1,1,ratio],[5,  1,1,0]]*2+
                 [[10, 2,1,0],[10, 1,1,0]]*1+
                 [[10, 1,1,ratio],[10, 1,1,0]]*7+
                 [[40, 2,1,0]],
                num_classes=num_classes,use_se=use_se)
