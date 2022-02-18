# A simplify version for better understanding
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet
def rm_r_BasicBlock(block):
    block.eval()
    in_planes = block.conv1.in_channels
    mid_planes = in_planes + block.conv1.out_channels
    out_planes = block.conv2.out_channels

    #merge conv1 and bn1
    block.conv1=nn.utils.fuse_conv_bn_eval(block.conv1,block.bn1)
    #new conv1
    idconv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=block.stride, padding=1).eval()
    #origional channels
    idconv1.weight.data[in_planes:]=block.conv1.weight.data
    idconv1.bias.data[in_planes:]=block.conv1.bias.data
    #reserve input featuremaps with dirac initialized channels
    nn.init.dirac_(idconv1.weight.data[:in_planes])
    nn.init.zeros_(idconv1.bias.data[:in_planes])

    #merge conv2 and bn2
    block.conv2=nn.utils.fuse_conv_bn_eval(block.conv2,block.bn2)
    #new conv
    idconv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1).eval()
    #origional channels
    idconv2.weight.data[:,in_planes:]=block.conv2.weight.data
    idconv2.bias.data=block.conv2.bias.data
    #merge input featuremaps to output featuremaps
    if in_planes==out_planes:
        nn.init.dirac_(idconv2.weight.data[:,:in_planes])
    else:
        #if there are a downsample layer
        downsample=nn.utils.fuse_conv_bn_eval(block.downsample[0],block.downsample[1])
        #conv1*1 -> conv3*3
        idconv2.weight.data[:,:in_planes]=F.pad(downsample.weight.data, [1, 1, 1, 1])
        idconv2.bias.data+=downsample.bias.data
    return nn.Sequential(*[idconv1,block.relu,idconv2,block.relu])

def resnet_to_vgg(model):
    model.layer1=nn.Sequential(*[rm_r_BasicBlock(block) for block in model.layer1])
    model.layer2=nn.Sequential(*[rm_r_BasicBlock(block) for block in model.layer2])
    model.layer3=nn.Sequential(*[rm_r_BasicBlock(block) for block in model.layer3])
    model.layer4=nn.Sequential(*[rm_r_BasicBlock(block) for block in model.layer4])
    
model=resnet.resnet18()
x=torch.randn(2,3,224,224)
model(x)
print(model.eval()(x))
resnet_to_vgg(model)
print(model.eval()(x))
