#a simplify version for better understing

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models.mobilenetv2 import InvertedResidual, mobilenet_v2

def rm_r_InvertedResidual(block):    
    block.eval()
    inp = block.conv[0][0].in_channels
    mid = inp+block.conv[0][0].out_channels
    oup = block.conv[2].out_channels

    conv1=nn.utils.fuse_conv_bn_eval(block.conv[0][0],block.conv[0][1])
    idconv1 = nn.Conv2d(inp, mid, kernel_size=1).eval()
    idrelu1 = nn.PReLU(mid)

    nn.init.dirac_(idconv1.weight.data[:inp])
    nn.init.zeros_(idconv1.bias.data[:inp])
    torch.nn.init.ones_(idrelu1.weight.data[:inp])
    
    idconv1.weight.data[inp:]=conv1.weight.data
    idconv1.bias.data[inp:]=conv1.bias.data
    torch.nn.init.zeros_(idrelu1.weight.data[inp:])

    conv2=nn.utils.fuse_conv_bn_eval(block.conv[1][0],block.conv[1][1])
    idconv2 = nn.Conv2d(mid, mid, kernel_size=3, stride=block.stride, padding=1,groups=mid).eval()
    idrelu2 = nn.PReLU(mid)
    
    nn.init.dirac_(idconv2.weight.data[:inp],groups=inp)
    nn.init.zeros_(idconv2.bias.data[:inp])
    torch.nn.init.ones_(idrelu2.weight.data[:inp])
    
    idconv2.weight.data[inp:]=conv2.weight.data
    idconv2.bias.data[inp:]=conv2.bias.data
    torch.nn.init.zeros_(idrelu2.weight.data[inp:])
    
    conv3=nn.utils.fuse_conv_bn_eval(block.conv[2],block.conv[3])
    idconv3 = nn.Conv2d(mid, oup, kernel_size=1).eval()
    
    nn.init.dirac_(idconv3.weight.data[:,:inp])
    idconv3.weight.data[:,inp:]=conv3.weight.data
    idconv3.bias.data=conv3.bias.data
    
    return [idconv1,idrelu1,idconv2,idrelu2,idconv3]

def mobilenetv2_to_mobilenetv1(model):
    features=[]
    for m in model.features:
        if isinstance(m,InvertedResidual)and m.use_res_connect:
                features+=rm_r_InvertedResidual(m)
        else:
            for mm in m.modules():
                if not list(mm.children()):
                    features.append(mm)
    model.features=nn.Sequential(*features)
    return model
    
model=mobilenet_v2()
x=torch.randn(1,3,224,224)
model(x)
print(model.eval()(x))
print(mobilenetv2_to_mobilenetv1(model).eval()(x))
