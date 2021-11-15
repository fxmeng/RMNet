import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, planes):
        super(ResBlock, self).__init__()
        self.in_planes = planes
        self.out_planes = planes
        self.mid_planes = planes*2
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        return out+x
    
    def deploy(self):
        idconv0 = nn.Conv2d(self.in_planes, self.mid_planes, kernel_size=1, bias=False).eval()
        nn.init.dirac_(idconv0.weight.data[:self.out_planes])
        nn.init.dirac_(idconv0.weight.data[self.out_planes:])
        
        
        idbn1=nn.BatchNorm2d(self.mid_planes)
        bn_var_sqrt=torch.sqrt(self.bn1.running_var + self.bn1.eps)
        idbn1.weight.data[:self.out_planes]=bn_var_sqrt
        idbn1.bias.data[:self.out_planes]=self.bn1.running_mean
        idbn1.running_mean.data[:self.out_planes]=self.bn1.running_mean
        idbn1.running_var.data[:self.out_planes]=self.bn1.running_var
        
        idbn1.weight.data[self.out_planes:]=self.bn1.weight.data
        idbn1.bias.data[self.out_planes:]=self.bn1.bias.data
        idbn1.running_mean.data[self.out_planes:]=self.bn1.running_mean
        idbn1.running_var.data[self.out_planes:]=self.bn1.running_var
        
        self.relu1 = nn.PReLU(self.mid_planes)
        torch.nn.init.ones_(self.relu1.weight.data[:self.out_planes])
        torch.nn.init.zeros_(self.relu1.weight.data[self.out_planes:])


        idconv1 = nn.Conv2d(self.mid_planes, self.mid_planes, kernel_size=3, padding=1, bias=False).eval()
        nn.init.dirac_(idconv1.weight.data[:self.out_planes])
        torch.nn.init.zeros_(idconv1.weight.data[self.out_planes:][:,:self.out_planes])
        idconv1.weight.data[self.out_planes:][:,self.out_planes:]=self.conv1.weight.data
        
        
        idbn2=nn.BatchNorm2d(self.mid_planes)
        idbn2.weight.data[:self.out_planes]=idbn1.weight.data[:self.out_planes]
        idbn2.bias.data[:self.out_planes]=idbn1.bias.data[:self.out_planes]
        idbn2.running_mean.data[:self.out_planes]=idbn1.running_mean.data[:self.out_planes]
        idbn2.running_var.data[:self.out_planes]=idbn1.running_var.data[:self.out_planes]
        
        idbn2.weight.data[self.out_planes:]=self.bn2.weight.data
        idbn2.bias.data[self.out_planes:]=self.bn2.bias.data
        idbn2.running_mean.data[self.out_planes:]=self.bn2.running_mean
        idbn2.running_var.data[self.out_planes:]=self.bn2.running_var
        
        self.relu2 = nn.PReLU(self.mid_planes)
        torch.nn.init.ones_(self.relu2.weight.data[:self.out_planes])
        torch.nn.init.zeros_(self.relu2.weight.data[self.out_planes:])
        
        
        idconv2 = nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3, stride=1, padding=1, bias=False).eval()
        nn.init.dirac_(idconv2.weight.data[:,:self.out_planes])
        idconv2.weight.data[:,self.out_planes:]=self.conv2.weight
        
        return [idconv0, idbn1, self.relu1, idconv1, idbn2, self.relu2, idconv2]
    
class ResChannel(nn.Module):
    def __init__(self, planes):
        super(ResChannel, self).__init__()
        self.in_planes = planes
        self.out_planes = planes*2
        self.mid_planes = planes*3

        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.running = nn.BatchNorm2d(self.in_planes,affine=False)
        self.downsample = nn.Conv2d(self.in_planes, self.out_planes, kernel_size=1, stride=2, bias=False)
        self.conv1 = nn.Conv2d(self.in_planes, self.out_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.out_planes, self.out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        self.running(out)
        shortcut = self.downsample(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        return out+shortcut
    
    def deploy(self):
        idconv1 = nn.Conv2d(self.in_planes, self.mid_planes, kernel_size=3, stride=2, padding=1, bias=False).eval()
        nn.init.dirac_(idconv1.weight.data[:self.in_planes])
        idconv1.weight.data[self.in_planes:]=self.conv1.weight.data
        
        idbn2=nn.BatchNorm2d(self.mid_planes)
        bn_var_sqrt=torch.sqrt(self.running.running_var + self.running.eps)
        idbn2.weight.data[:self.in_planes]=bn_var_sqrt
        idbn2.bias.data[:self.in_planes]=self.running.running_mean
        idbn2.running_mean.data[:self.in_planes]=self.running.running_mean
        idbn2.running_var.data[:self.in_planes]=self.running.running_var
        
        idbn2.weight.data[self.in_planes:]=self.bn2.weight.data
        idbn2.bias.data[self.in_planes:]=self.bn2.bias.data
        idbn2.running_mean.data[self.in_planes:]=self.bn2.running_mean
        idbn2.running_var.data[self.in_planes:]=self.bn2.running_var
 
        idconv2 = nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3, stride=1, padding=1, bias=False).eval()
        idconv2.weight.data[:,:self.in_planes]=F.pad(self.downsample.weight.data, [1, 1, 1, 1])
        idconv2.weight.data[:,self.in_planes:]=self.conv2.weight.data
        return [self.bn1, self.relu1, idconv1, idbn2, self.relu2, idconv2]

class PreActResVGG(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super(PreActResVGG, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.flat = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        layers = []
        if stride==2:
            layers.append(ResChannel(self.in_planes))
        else:
            layers.append(ResBlock(self.in_planes))
        self.in_planes = planes
        for i in range(num_blocks-1):
            layers.append(ResBlock(self.in_planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.gap(out)
        out = self.flat(out)
        out = self.fc(out)
        return out

    def rm(self):
        def foo(net):
            global blocks
            childrens = list(net.children())
            if isinstance(net, ResBlock)or isinstance(net, ResChannel):
                blocks+=net.deploy()
            elif not childrens:
                blocks+=[net]
            else:
                for c in childrens:
                    foo(c)
        global blocks
        blocks =[]
        foo(self)
        return nn.Sequential(*blocks)
    
    def deploy(self):
        model=self.rm()
        blocks=[]
        c11=None
        for m in model[::-1]:
            if isinstance(m,nn.Conv2d):
                if m.kernel_size==(1,1):
                    c11=m
                else:
                    if c11 is not None:
                        c31=nn.Conv2d(m.in_channels,c11.out_channels,3,stride=m.stride, padding=1,bias=False)
                        c31.weight.data=(c11.weight.data.view(c11.out_channels,c11.in_channels)@m.weight.data.view(m.out_channels,-1)).view(c11.out_channels,m.in_channels,3,3)
                        c11=None
                        blocks.append(c31)
                    else:
                        blocks.append(m)
            else:
                blocks.append(m)
        return nn.Sequential(*blocks[::-1])


def preactresvgg18(num_classes=10):
    return PreActResVGG([2, 2, 2, 2], num_classes=num_classes)

def preactresvgg34(num_classes=10):
    return PreActResVGG([3, 4, 6, 3], num_classes=num_classes)
