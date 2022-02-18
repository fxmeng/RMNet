# A simplify version for better understanding

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models.mobilenetv2 import InvertedResidual, mobilenet_v2

def rm_r_InvertedResidual(block):
    inp = block.conv[0][0].in_channels
    mid = inp+block.conv[0][0].out_channels
    oup = block.conv[2].out_channels

    #merge conv1 and bn1
    conv1=nn.utils.fuse_conv_bn_eval(block.conv[0][0],block.conv[0][1])
    #new conv1
    idconv1 = nn.Conv2d(inp, mid, kernel_size=1).eval()
    idrelu1 = nn.PReLU(mid)
    #origional channels
    idconv1.weight.data[inp:]=conv1.weight.data
    idconv1.bias.data[inp:]=conv1.bias.data
    torch.nn.init.zeros_(idrelu1.weight.data[inp:])
    #reserve input featuremaps with dirac initialized channels
    nn.init.dirac_(idconv1.weight.data[:inp])
    nn.init.zeros_(idconv1.bias.data[:inp])
    torch.nn.init.ones_(idrelu1.weight.data[:inp])

    #merge conv2 and bn2
    conv2=nn.utils.fuse_conv_bn_eval(block.conv[1][0],block.conv[1][1])
    #new conv2
    idconv2 = nn.Conv2d(mid, mid, kernel_size=3, stride=block.stride, padding=1,groups=mid).eval()
    idrelu2 = nn.PReLU(mid)
    #origional channels
    idconv2.weight.data[inp:]=conv2.weight.data
    idconv2.bias.data[inp:]=conv2.bias.data
    torch.nn.init.zeros_(idrelu2.weight.data[inp:])
    #reserve input featuremaps with dirac initialized channels
    nn.init.dirac_(idconv2.weight.data[:inp],groups=inp)
    nn.init.zeros_(idconv2.bias.data[:inp])
    torch.nn.init.ones_(idrelu2.weight.data[:inp])
    
    #merge conv3 and bn3
    conv3=nn.utils.fuse_conv_bn_eval(block.conv[2],block.conv[3])
    #new conv3
    idconv3 = nn.Conv2d(mid, oup, kernel_size=1).eval()
    #origional channels
    idconv3.weight.data[:,inp:]=conv3.weight.data
    idconv3.bias.data=conv3.bias.data
    #merge input featuremaps to output featuremaps
    nn.init.dirac_(idconv3.weight.data[:,:inp])
    
    return [idconv1,idrelu1,idconv2,idrelu2,idconv3]

def fuse_conv1_conv2(conv1,conv2):
    inp=conv1.in_channels
    mid=conv1.out_channels
    oup=conv2.out_channels
    fused_conv=nn.Conv2d(inp,oup,1)
    fused_conv.weight.data=(conv2.weight.data.view(oup,mid)@conv1.weight.data.view(mid,-1)).view(oup,inp,1,1)
    fused_conv.bias.data=conv2.bias.data+conv2.weight.data.view(oup,mid)@conv1.bias.data
    return fused_conv

def mobilenetv2_to_mobilenetv1(model):
    model.eval()
    features=[]
    for m in model.features:
        if isinstance(m,InvertedResidual)and m.use_res_connect:
                features+=rm_r_InvertedResidual(m)
        else:
            for mm in m.modules():
                if not list(mm.children()):
                    #fuse conv and bn
                    if isinstance(mm,nn.Conv2d):
                        conv=mm
                        continue
                    elif isinstance(mm,nn.BatchNorm2d):
                        mm=nn.utils.fuse_conv_bn_eval(conv,mm)
                    features.append(mm)
                    
    #fuse consecutive convolutional layers
    new_features=[features[0]]
    for m in features[1:]:
        if isinstance(m,nn.Conv2d)and isinstance(new_features[-1],nn.Conv2d):
            new_features[-1]=fuse_conv1_conv2(new_features[-1],m)
        else:
            new_features.append(m)
    model.features=nn.Sequential(*new_features)
    return model

model=mobilenet_v2()
x=torch.randn(2,3,224,224)
model(x)
print(model.eval()(x))
mobilenetv2_to_mobilenetv1(model)
model(x)
print(model.eval()(x))
