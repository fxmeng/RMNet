import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class RepBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):

        super(RepBlock, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride

        self.conv33 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn33 = nn.BatchNorm2d(out_planes)
        self.conv11 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn11 = nn.BatchNorm2d(out_planes)
        if self.in_planes == self.out_planes and self.stride == 1:
            self.bn00 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = self.bn33(self.conv33(x))
        out += self.bn11(self.conv11(x))
        if self.in_planes == self.out_planes and self.stride == 1:
            out += self.bn00(x)
        return F.relu(out)

    def deploy(self):
        self.eval()
        conv33_bn33 = torch.nn.utils.fuse_conv_bn_eval(self.conv33, self.bn33).eval()
        conv11_bn11 = torch.nn.utils.fuse_conv_bn_eval(self.conv11, self.bn11).eval()
        conv33_bn33.weight.data += F.pad(conv11_bn11.weight.data, [1, 1, 1, 1])
        conv33_bn33.bias.data += conv11_bn11.bias.data
        if self.in_planes == self.out_planes and self.stride == 1:
            conv00 = nn.Conv2d(self.in_planes, self.out_planes, kernel_size=3, padding=1, bias=False).eval()
            nn.init.dirac_(conv00.weight.data)
            conv00_bn00 = torch.nn.utils.fuse_conv_bn_eval(conv00, self.bn00)
            conv33_bn33.weight.data += conv00_bn00.weight.data
            conv33_bn33.bias.data += conv00_bn00.bias.data
        return [conv33_bn33,nn.ReLU(True)]

class RMBlock(nn.Module):
    def __init__(self, planes, ratio=0.5):

        super(RMBlock, self).__init__()
        self.planes = planes
        self.rmplanes=int(planes*ratio)
        
        self.conv33 = nn.Conv2d(planes, self.planes-self.rmplanes, kernel_size=3, padding=1, bias=False)
        self.bn33 = nn.BatchNorm2d(self.planes-self.rmplanes)
        self.conv11 = nn.Conv2d(planes, self.planes-self.rmplanes, kernel_size=1, padding=0, bias=False)
        self.bn11 = nn.BatchNorm2d(self.planes-self.rmplanes)
        self.bn00 = nn.BatchNorm2d(self.planes-self.rmplanes)

    def forward(self, x):
        out = self.bn33(self.conv33(x))
        out += self.bn11(self.conv11(x))
        out += self.bn00(x[:,self.rmplanes:])
        return F.relu(torch.cat([x[:,:self.rmplanes],out],dim=1))
    
    def deploy(self):
        self.eval()
        conv33=nn.utils.fuse_conv_bn_eval(self.conv33,self.bn33)
        conv11=nn.utils.fuse_conv_bn_eval(self.conv11,self.bn11)
        conv00=nn.Conv2d(self.planes,self.planes-self.rmplanes,kernel_size=3,padding=1,bias=False).eval()
        nn.init.zeros_(conv00.weight.data[:,:self.rmplanes])
        nn.init.dirac_(conv00.weight.data[:,self.rmplanes:])
        conv00=nn.utils.fuse_conv_bn_eval(conv00,self.bn00)
        conv3=nn.Conv2d(self.planes,self.planes,kernel_size=3,padding=1)
        conv1=nn.Conv2d(self.planes,self.planes,kernel_size=1)
        conv0=nn.Conv2d(self.planes,self.planes,kernel_size=3,padding=1)
        nn.init.zeros_(conv3.weight.data[:self.rmplanes])
        nn.init.zeros_(conv1.weight.data[:self.rmplanes])
        nn.init.dirac_(conv0.weight.data[:self.rmplanes])
        nn.init.zeros_(conv3.bias.data[:self.rmplanes])
        nn.init.zeros_(conv1.bias.data[:self.rmplanes])
        nn.init.zeros_(conv0.bias.data[:self.rmplanes])
        conv3.weight.data[self.rmplanes:]=conv33.weight.data
        conv1.weight.data[self.rmplanes:]=conv11.weight.data
        conv0.weight.data[self.rmplanes:]=conv00.weight.data
        conv3.bias.data[self.rmplanes:]=conv33.bias.data
        conv1.bias.data[self.rmplanes:]=conv11.bias.data
        conv0.bias.data[self.rmplanes:]=conv00.bias.data
        
        conv3.weight.data += F.pad(conv1.weight.data, [1, 1, 1, 1])
        conv3.bias.data += conv1.bias.data
        conv3.weight.data += conv0.weight.data
        conv3.bias.data += conv0.bias.data
        return [conv3,nn.ReLU(True)]
        
    
class RMRep(nn.Module):
    def __init__(self, num_blocks, num_classes=1000, base_wide=64,ratio=0.5):
        super(RMRep, self).__init__()
        feature=[]
        in_planes=3
        for b,t,s,n in num_blocks:
            out_planes=t*base_wide
            for i in range(n):
                if b=='rm_rep':
                    feature.append(RMBlock(out_planes,ratio))
                feature.append(RepBlock(in_planes,out_planes,s))
                in_planes=out_planes
                
        self.feature=nn.Sequential(*feature)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.flat = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(base_wide*num_blocks[-1][1], num_classes)

    def forward(self, x):
        out = self.feature(x)
        out = self.gap(out)
        out = self.flat(out)
        out = self.fc(out)
        return out
    
    def deploy(self):
        blocks=[]
        for m in self.feature:
            if isinstance(m,RepBlock) or isinstance(m,RMBlock):
                blocks+=m.deploy()
        blocks.append(self.gap)
        blocks.append(self.flat)
        blocks.append(self.fc)
        return nn.Sequential(*blocks)



def repvgg_21(num_classes=1000,depth=2):     
    return RMRep([['rep',1,2,2] if num_classes==1000 else ['rep',1,1,1],
                   ['rm_rep',1,1,depth],
                   ['rep_rep',2,2,1],
                   ['rm_rep',2,1,depth],
                   ['rep',4,2,1],
                   ['rm_rep',4,1,depth],
                   ['rep',8,2,1],
                   ['rm_rep',8,1,depth]],
                   num_classes=num_classes,ratio=0)

def repvgg_37(num_classes=1000,depth=4):     
    return RMRep([['rep',1,2,2] if num_classes==1000 else ['rep',1,1,1],
                   ['rm_rep',1,1,depth],
                   ['rep_rep',2,2,1],
                   ['rm_rep',2,1,depth],
                   ['rep',4,2,1],
                   ['rm_rep',4,1,depth],
                   ['rep',8,2,1],
                   ['rm_rep',8,1,depth]],
                   num_classes=num_classes,ratio=0)

def repvgg_69(num_classes=1000,depth=8):     
    return RMRep([['rep',1,2,2] if num_classes==1000 else ['rep',1,1,1],
                   ['rm_rep',1,1,depth],
                   ['rep_rep',2,2,1],
                   ['rm_rep',2,1,depth],
                   ['rep',4,2,1],
                   ['rm_rep',4,1,depth],
                   ['rep',8,2,1],
                   ['rm_rep',8,1,depth]],
                   num_classes=num_classes,ratio=0)
            
def repvgg_133(num_classes=1000,depth=16):     
    return RMRep([['rep',1,2,2] if num_classes==1000 else ['rep',1,1,1],
                   ['rm_rep',1,1,depth],
                   ['rep_rep',2,2,1],
                   ['rm_rep',2,1,depth],
                   ['rep',4,2,1],
                   ['rm_rep',4,1,depth],
                   ['rep',8,2,1],
                   ['rm_rep',8,1,depth]],
                   num_classes=num_classes,ratio=0)



def rmrep_21(num_classes=1000,depth=2):     
    return RMRep([['rep',1,2,2] if num_classes==1000 else ['rep',1,1,1],
                   ['rm_rep',1,1,depth],
                   ['rep_rep',2,2,1],
                   ['rm_rep',2,1,depth],
                   ['rep',4,2,1],
                   ['rm_rep',4,1,depth],
                   ['rep',8,2,1],
                   ['rm_rep',8,1,depth]],
                   num_classes=num_classes,ratio=0.25)

def rmrep_37(num_classes=1000,depth=4):     
    return RMRep([['rep',1,2,2] if num_classes==1000 else ['rep',1,1,1],
                   ['rm_rep',1,1,depth],
                   ['rep_rep',2,2,1],
                   ['rm_rep',2,1,depth],
                   ['rep',4,2,1],
                   ['rm_rep',4,1,depth],
                   ['rep',8,2,1],
                   ['rm_rep',8,1,depth]],
                   num_classes=num_classes,ratio=0.25)

def rmrep_69(num_classes=1000,depth=8):     
    return RMRep([['rep',1,2,2] if num_classes==1000 else ['rep',1,1,1],
                   ['rm_rep',1,1,depth],
                   ['rep_rep',2,2,1],
                   ['rm_rep',2,1,depth],
                   ['rep',4,2,1],
                   ['rm_rep',4,1,depth],
                   ['rep',8,2,1],
                   ['rm_rep',8,1,depth]],
                   num_classes=num_classes,ratio=0.5)
            
def rmrep_133(num_classes=1000,depth=16):     
    return RMRep([['rep',1,2,2] if num_classes==1000 else ['rep',1,1,1],
                   ['rm_rep',1,1,depth],
                   ['rep_rep',2,2,1],
                   ['rm_rep',2,1,depth],
                   ['rep',4,2,1],
                   ['rm_rep',4,1,depth],
                   ['rep',8,2,1],
                   ['rm_rep',8,1,depth]],
                   num_classes=num_classes,ratio=0.75)


