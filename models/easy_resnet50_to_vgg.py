import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet

def resnet50_to_vgg(model):
    def rm_r_Bottleneck(block):
        block.eval()
        in_planes = block.conv1.in_channels
        mid_planes = in_planes + block.conv1.out_channels
        out_planes = block.conv3.out_channels

        #merge conv1 and bn1
        block.conv1=nn.utils.fuse_conv_bn_eval(block.conv1,block.bn1)
        #new conv1
        idconv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1).eval()
        #origional channels
        idconv1.weight.data[in_planes:]=block.conv1.weight.data
        idconv1.bias.data[in_planes:]=block.conv1.bias.data
        #reserve input featuremaps with dirac initialized channels
        nn.init.dirac_(idconv1.weight.data[:in_planes])
        nn.init.zeros_(idconv1.bias.data[:in_planes])


        #merge conv2 and bn2
        block.conv2=nn.utils.fuse_conv_bn_eval(block.conv2,block.bn2)
        #new conv2
        idconv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=block.stride, padding=1).eval()
        #origional channels
        idconv2.weight.data[in_planes:][:,in_planes:]=block.conv2.weight.data
        nn.init.zeros_(idconv2.weight.data[in_planes:][:,:in_planes])
        idconv2.bias.data[in_planes:]=block.conv2.bias.data
        #reserve input featuremaps with dirac initialized channels
        nn.init.dirac_(idconv2.weight.data[:in_planes])
        nn.init.zeros_(idconv2.bias.data[:in_planes])

        #merge conv3 and bn3
        block.conv3=nn.utils.fuse_conv_bn_eval(block.conv3,block.bn3)
        #new conv3
        idconv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1).eval()
        #origional channels
        idconv3.weight.data[:,in_planes:]=block.conv3.weight.data
        idconv3.bias.data=block.conv3.bias.data
        #merge input featuremaps to output featuremaps
        if in_planes==out_planes:
            nn.init.dirac_(idconv3.weight.data[:,:in_planes])
        else:
            #if there are a downsample layer
            downsample=nn.utils.fuse_conv_bn_eval(block.downsample[0],block.downsample[1])
            idconv3.weight.data[:,:in_planes]=downsample.weight.data
            idconv3.bias.data+=downsample.bias.data
        return nn.Sequential(*[idconv1,block.relu,idconv2,block.relu,idconv3,block.relu])

    model.layer1=nn.Sequential(*[rm_r_Bottleneck(block) for block in model.layer1])
    model.layer2=nn.Sequential(*[rm_r_Bottleneck(block) for block in model.layer2])
    model.layer3=nn.Sequential(*[rm_r_Bottleneck(block) for block in model.layer3])
    model.layer4=nn.Sequential(*[rm_r_Bottleneck(block) for block in model.layer4])
    
model=resnet.resnet50()
x=torch.randn(2,3,224,224)
model(x)
print(model.eval()(x))
resnet50_to_vgg(model)
print(model.eval()(x))
