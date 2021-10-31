import torch
import torch.nn as nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(self, in_planes, mid_planes, out_planes, stride=1):
        super(ResBlock, self).__init__()

        assert mid_planes > in_planes

        self.in_planes = in_planes
        self.mid_planes = mid_planes - out_planes +in_planes
        self.out_planes = out_planes
        self.stride = stride

        self.conv1 = nn.Conv2d(in_planes, self.mid_planes - in_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.mid_planes - in_planes)
        
        self.conv2 = nn.Conv2d(self.mid_planes - in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample=nn.Sequential()
        if self.in_planes != self.out_planes or self.stride != 1:
            self.downsample=nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes))
        self.running1 = nn.BatchNorm2d(in_planes,affine=False)
        self.running2 = nn.BatchNorm2d(out_planes,affine=False)
        
    def forward(self, x):
        if self.in_planes == self.out_planes and self.stride == 1:
            self.running1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        self.running2(out)
        return self.relu(out)
    
    def deploy(self, merge_bn=False):
        idconv1 = nn.Conv2d(self.in_planes, self.mid_planes, kernel_size=3, stride=self.stride, padding=1, bias=False).eval()
        idbn1=nn.BatchNorm2d(self.mid_planes).eval()
        
        nn.init.dirac_(idconv1.weight.data[:self.in_planes])
        bn_var_sqrt=torch.sqrt(self.running1.running_var + self.running1.eps)
        idbn1.weight.data[:self.in_planes]=bn_var_sqrt
        idbn1.bias.data[:self.in_planes]=self.running1.running_mean
        idbn1.running_mean.data[:self.in_planes]=self.running1.running_mean
        idbn1.running_var.data[:self.in_planes]=self.running1.running_var
        
        idconv1.weight.data[self.in_planes:]=self.conv1.weight.data
        idbn1.weight.data[self.in_planes:]=self.bn1.weight.data
        idbn1.bias.data[self.in_planes:]=self.bn1.bias.data
        idbn1.running_mean.data[self.in_planes:]=self.bn1.running_mean
        idbn1.running_var.data[self.in_planes:]=self.bn1.running_var
        
        idconv2 = nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3, stride=1, padding=1, bias=False).eval()
        idbn2=nn.BatchNorm2d(self.out_planes).eval()
        downsample_bias=0
        if self.in_planes==self.out_planes:
            nn.init.dirac_(idconv2.weight.data[:,:self.in_planes])
        else:
            idconv2.weight.data[:,:self.in_planes],downsample_bias=self.fuse(F.pad(self.downsample[0].weight.data, [1, 1, 1, 1]),self.downsample[1].running_mean,self.downsample[1].running_var,self.downsample[1].weight,self.downsample[1].bias,self.downsample[1].eps)

        idconv2.weight.data[:,self.in_planes:],bias=self.fuse(self.conv2.weight,self.bn2.running_mean,self.bn2.running_var,self.bn2.weight,self.bn2.bias,self.bn2.eps)
        
        bn_var_sqrt=torch.sqrt(self.running2.running_var + self.running2.eps)
        idbn2.weight.data=bn_var_sqrt
        idbn2.bias.data=self.running2.running_mean
        idbn2.running_mean.data=self.running2.running_mean+bias+downsample_bias
        idbn2.running_var.data=self.running2.running_var
        
        if merge_bn:
            return [torch.nn.utils.fuse_conv_bn_eval(idconv1,idbn1),self.relu,torch.nn.utils.fuse_conv_bn_eval(idconv2,idbn2),self.relu]
        else:
            return [idconv1,idbn1,self.relu,idconv2,idbn2,self.relu]


    def fuse(self,conv_w, bn_rm, bn_rv,bn_w,bn_b, eps):
        bn_var_rsqrt = torch.rsqrt(bn_rv + eps)
        conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
        conv_b = bn_rm * bn_var_rsqrt * bn_w-bn_b
        return conv_w,conv_b
    
class RMNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,base_wide=64):
        super(RMNet, self).__init__()
        self.in_planes = base_wide
        self.conv1 = nn.Conv2d(3, base_wide, kernel_size=7 if num_classes==1000 else 3, stride=2 if num_classes==1000 else 1, padding=3 if num_classes==1000 else 1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_wide)
        self.relu = nn.ReLU(inplace=True)
        if num_classes==1000:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, base_wide, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_wide*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_wide*4, num_blocks[2], stride=2)
        self.layer4 = None
        if len(num_blocks)==4:
            self.layer4 = self._make_layer(block, base_wide*8, num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.flat = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(self.in_planes, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes*2, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        if self.fc.out_features==1000:
            out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.layer4 is not None:
            out = self.layer4(out)
        out = self.gap(out)
        out = self.flat(out)
        out = self.fc(out)
        return out

    def deploy(self, merge_bn=False):
        def foo(net):
            global blocks
            childrens = list(net.children())
            if isinstance(net, ResBlock):
                blocks+=net.deploy(merge_bn)
            elif not childrens:
                if isinstance(net,nn.BatchNorm2d) and isinstance(blocks[-1],nn.Conv2d):
                    blocks[-1]=torch.nn.utils.fuse_conv_bn_eval(blocks[-1],net)
                else:
                    blocks+=[net]
            else:
                for c in childrens:
                    foo(c)
        global blocks
        
        blocks =[]
        foo(self.eval())
        return nn.Sequential(*blocks)

def rmnet18(num_classes=1000):
    return RMNet(ResBlock, [2, 2, 2, 2], num_classes=num_classes)

def rmnet34(num_classes=1000):
    return RMNet(ResBlock, [3, 4, 6, 3], num_classes=num_classes)

def rmnet20(num_classes=10):
    return RMNet(ResBlock, [3, 3, 3],num_classes=num_classes,base_wide=16)

def rmnet32(num_classes=10):
    return RMNet(ResBlock, [5, 5, 5],num_classes=num_classes,base_wide=16)

def rmnet44(num_classes=10):
    return RMNet(ResBlock, [7, 7, 7],num_classes=num_classes,base_wide=16)

def rmnet56(num_classes=10):
    return RMNet(ResBlock, [9, 9, 9],num_classes=num_classes,base_wide=16)

def rmnet110(num_classes=10):
    return RMNet(ResBlock, [18, 18, 18],num_classes=num_classes,base_wide=16)

def rmnet1202(num_classes=10):
    return RMNet(ResBlock, [200, 200, 200],num_classes=num_classes,base_wide=16)
