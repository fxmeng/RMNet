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
        #self.running = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn33(self.conv33(x))
        out += self.bn11(self.conv11(x))
        #self.running(out)
        return self.relu(out)

    def deploy(self, merge_bn=False):
        self.eval()
        conv33_bn33 = torch.nn.utils.fuse_conv_bn_eval(self.conv33, self.bn33).eval()
        conv11_bn11 = torch.nn.utils.fuse_conv_bn_eval(self.conv11, self.bn11).eval()
        conv33_bn33.weight.data += F.pad(conv11_bn11.weight.data, [1, 1, 1, 1])
        conv33_bn33.bias.data += conv11_bn11.bias.data

        #self.running.weight.data = torch.sqrt(self.running.running_var + self.running.eps)
        #self.running.bias.data = self.running.running_mean
        #if merge_bn:
        return [conv33_bn33, self.relu]
        #else:
        #    return [conv33_bn33, self.running, self.relu]


class RMBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, expand_ratio=2, cpg=1):
        super(RMBlock, self).__init__()
        self.in_planes = in_planes
        self.mid_planes = out_planes * expand_ratio - out_planes
        self.out_planes = out_planes
        self.stride = stride
        self.cpg = cpg
        assert self.mid_planes % cpg == 0 and (self.mid_planes + self.in_planes) % cpg == 0
        self.groups = self.mid_planes // cpg

        self.conv1 = nn.Conv2d(in_planes, self.mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.mid_planes)
        self.conv2 = nn.Conv2d(self.mid_planes, self.mid_planes, kernel_size=3, stride=stride, padding=1, groups=self.groups, bias=False)
        self.bn2 = nn.BatchNorm2d(self.mid_planes)
        self.conv3 = nn.Conv2d(self.mid_planes, out_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential()
        if self.stride != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )
        self.running1 = nn.BatchNorm2d(self.in_planes, affine=False)
        self.running2 = nn.BatchNorm2d(self.out_planes, affine=False)

    def forward(self, x):
        self.running1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        self.running2(out)
        return self.relu(out)

    def deploy(self, merge_bn=False):
        self.mid_planes = self.conv2.in_channels + self.in_planes
        self.groups = self.mid_planes // self.cpg
        idconv1 = nn.Conv2d(self.in_planes, self.mid_planes, kernel_size=1, bias=False).eval()
        idbn1 = nn.BatchNorm2d(self.mid_planes).eval()

        nn.init.dirac_(idconv1.weight.data[:self.in_planes])
        bn_var_sqrt = torch.sqrt(self.running1.running_var + self.running1.eps)
        idbn1.weight.data[:self.in_planes] = bn_var_sqrt
        idbn1.bias.data[:self.in_planes] = self.running1.running_mean
        idbn1.running_mean.data[:self.in_planes] = self.running1.running_mean
        idbn1.running_var.data[:self.in_planes] = self.running1.running_var

        idconv1.weight.data[self.in_planes:] = self.conv1.weight.data
        idbn1.weight.data[self.in_planes:] = self.bn1.weight.data
        idbn1.bias.data[self.in_planes:] = self.bn1.bias.data
        idbn1.running_mean.data[self.in_planes:] = self.bn1.running_mean
        idbn1.running_var.data[self.in_planes:] = self.bn1.running_var

        idconv2 = nn.Conv2d(self.mid_planes, self.mid_planes, kernel_size=3, stride=self.stride, padding=1, groups=self.groups, bias=False).eval()
        idbn2 = nn.BatchNorm2d(self.mid_planes).eval()

        idbn2.weight.data[:self.in_planes] = idbn1.weight.data[:self.in_planes]
        idbn2.bias.data[:self.in_planes] = idbn1.bias.data[:self.in_planes]
        idbn2.running_mean.data[:self.in_planes] = idbn1.running_mean.data[:self.in_planes]
        idbn2.running_var.data[:self.in_planes] = idbn1.running_var.data[:self.in_planes]
        nn.init.dirac_(idconv2.weight.data[:self.in_planes], groups=self.groups - self.conv2.groups)

        idconv2.weight.data[self.in_planes:] = self.conv2.weight.data
        idbn2.weight.data[self.in_planes:] = self.bn2.weight.data
        idbn2.bias.data[self.in_planes:] = self.bn2.bias.data
        idbn2.running_mean.data[self.in_planes:] = self.bn2.running_mean
        idbn2.running_var.data[self.in_planes:] = self.bn2.running_var

        idconv3 = nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=1, bias=False).eval()
        idbn3 = nn.BatchNorm2d(self.out_planes).eval()

        downsample_bias = 0
        if self.in_planes == self.out_planes and self.stride == 1:
            nn.init.dirac_(idconv3.weight.data[:, :self.in_planes])
        else:
            idconv3.weight.data[:, :self.in_planes], downsample_bias = self.fuse(self.downsample[0].weight, self.downsample[1].running_mean, self.downsample[1].running_var, self.downsample[1].weight, self.downsample[1].bias, self.downsample[1].eps)

        idconv3.weight.data[:, self.in_planes:], bias = self.fuse(self.conv3.weight, self.bn3.running_mean, self.bn3.running_var, self.bn3.weight, self.bn3.bias, self.bn3.eps)
        bn_var_sqrt = torch.sqrt(self.running2.running_var + self.running2.eps)
        idbn3.weight.data = bn_var_sqrt
        idbn3.bias.data = self.running2.running_mean
        idbn3.running_mean.data = self.running2.running_mean + bias + downsample_bias
        idbn3.running_var.data = self.running2.running_var

        if merge_bn:
            return [torch.nn.utils.fuse_conv_bn_eval(idconv1, idbn1), self.relu, torch.nn.utils.fuse_conv_bn_eval(idconv2, idbn2), self.relu, torch.nn.utils.fuse_conv_bn_eval(idconv3, idbn3), self.relu]
        else:
            return [idconv1, idbn1, self.relu, idconv2, idbn2, self.relu, idconv3, idbn3, self.relu]

    def fuse(self, conv_w, bn_rm, bn_rv, bn_w, bn_b, eps):
        bn_var_rsqrt = torch.rsqrt(bn_rv + eps)
        conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
        conv_b = bn_rm * bn_var_rsqrt * bn_w - bn_b
        return conv_w, conv_b


class RMNeXt(nn.Module):
    def __init__(self, num_blocks, num_classes=1000, base_wide=64, expand_ratio=2, cpg=1):
        super(RMNeXt, self).__init__()
        self.in_planes = min(64, base_wide)
        if num_classes==1000:
            self.layer0 = nn.Sequential(*[RepBlock(3, self.in_planes, stride=2), RepBlock(self.in_planes, self.in_planes, stride=2)])
        else:
            self.layer0 = RepBlock(3, self.in_planes, stride=1)
        self.layer1 = self._make_layer(base_wide, num_blocks[0], expand_ratio, cpg, stride=1)
        self.layer2 = self._make_layer(base_wide * 2, num_blocks[1], expand_ratio, cpg * 2, stride=2)
        self.layer3 = self._make_layer(base_wide * 4, num_blocks[2], expand_ratio, cpg * 4, stride=2)
        self.layer4 = self._make_layer(base_wide * 8, num_blocks[3], expand_ratio, cpg * 8, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.flat = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(self.in_planes, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, RMBlock):
                nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, planes, num_blocks, expand_ratio, cpg, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(RMBlock(self.in_planes, planes, stride, expand_ratio, cpg))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.gap(out)
        out = self.flat(out)
        out = self.fc(out)
        return out

    def deploy(self, merge_bn=False):
        def foo(net):
            global blocks
            childrens = list(net.children())
            if isinstance(net, RMBlock) or isinstance(net, RepBlock):
                blocks += net.deploy(merge_bn)
            elif not childrens:
                blocks += [net]
            else:
                for c in childrens:
                    foo(c)

        global blocks

        blocks = []
        foo(self.eval())
        return nn.Sequential(*blocks)


def rmnext41_64x5_g16(num_classes=1000):
    return RMNeXt([2, 3, 5, 3], num_classes=num_classes, base_wide=64, expand_ratio=5, cpg=16)


def rmnext50_64x5_g32(num_classes=1000):
    return RMNeXt([3, 4, 6, 3], num_classes=num_classes, base_wide=64, expand_ratio=5, cpg=32)


def rmnext50_64x6_g32(num_classes=1000):
    return RMNeXt([3, 4, 6, 3], num_classes=num_classes, base_wide=64, expand_ratio=6, cpg=32)


def rmnext101_64x6_g16(num_classes=1000):
    return RMNeXt([3, 4, 23, 3], num_classes=num_classes, base_wide=64, expand_ratio=6, cpg=16)


def rmnext152_64x6_g32(num_classes=1000):
    return RMNeXt([3, 8, 36, 3], num_classes=num_classes, base_wide=64, expand_ratio=6, cpg=32)
