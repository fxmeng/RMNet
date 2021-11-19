import torch
import torch.nn as nn
import torchvision
import math


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def fuse_cbcb(conv1,bn1,conv2,bn2):
    inp=conv1.in_channels
    mid=conv1.out_channels
    oup=conv2.out_channels
    conv1=torch.nn.utils.fuse_conv_bn_eval(conv1.eval(),bn1.eval())
    fused_conv=nn.Conv2d(inp,oup,1,bias=False)
    fused_conv.weight.data=(conv2.weight.data.view(oup,mid)@conv1.weight.data.view(mid,-1)).view(oup,inp,1,1)
    bn2.running_mean-=conv2.weight.data.view(oup,mid)@conv1.bias.data
    return fused_conv,bn2

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.in_planes=inp
        self.out_planes=oup
        
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.mid_planes=hidden_dim+ inp if self.use_res_connect else 0
        
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        if self.use_res_connect:
            self.running1 = nn.BatchNorm2d(self.in_planes,affine=False)
            self.running2 = nn.BatchNorm2d(self.out_planes,affine=False)

    def forward(self, x):
        if self.use_res_connect:
            self.running1(x)
            out=x + self.conv(x)
            self.running2(out)
            return out
        else:
            return self.conv(x)
        
    def deploy(self):
        if self.use_res_connect:
            idconv1 = nn.Conv2d(self.in_planes, self.mid_planes, kernel_size=1, bias=False).eval()
            idbn1=nn.BatchNorm2d(self.mid_planes).eval()

            nn.init.dirac_(idconv1.weight.data[:self.in_planes])
            bn_var_sqrt=torch.sqrt(self.running1.running_var + self.running1.eps)
            idbn1.weight.data[:self.out_planes]=bn_var_sqrt
            idbn1.bias.data[:self.out_planes]=self.running1.running_mean
            idbn1.running_mean.data[:self.out_planes]=self.running1.running_mean
            idbn1.running_var.data[:self.out_planes]=self.running1.running_var

            idconv1.weight.data[self.out_planes:]=self.conv[0].weight.data
            idbn1.weight.data[self.out_planes:]=self.conv[1].weight.data
            idbn1.bias.data[self.out_planes:]=self.conv[1].bias.data
            idbn1.running_mean.data[self.out_planes:]=self.conv[1].running_mean
            idbn1.running_var.data[self.out_planes:]=self.conv[1].running_var
            idrelu1 = nn.PReLU(self.mid_planes)
            torch.nn.init.ones_(idrelu1.weight.data[:self.in_planes])
            torch.nn.init.zeros_(idrelu1.weight.data[self.in_planes:])



            idconv2 = nn.Conv2d(self.mid_planes, self.mid_planes, kernel_size=3, stride=self.stride, padding=1,groups=self.mid_planes, bias=False).eval()
            idbn2=nn.BatchNorm2d(self.mid_planes).eval()

            nn.init.dirac_(idconv2.weight.data[:self.in_planes],groups=self.in_planes)
            idbn2.weight.data[:self.out_planes]=idbn1.weight.data[:self.out_planes]
            idbn2.bias.data[:self.out_planes]=idbn1.bias.data[:self.out_planes]
            idbn2.running_mean.data[:self.out_planes]=idbn1.running_mean.data[:self.out_planes]
            idbn2.running_var.data[:self.out_planes]=idbn1.running_var.data[:self.out_planes]

            idconv2.weight.data[self.out_planes:]=self.conv[3].weight.data
            idbn2.weight.data[self.out_planes:]=self.conv[4].weight.data
            idbn2.bias.data[self.out_planes:]=self.conv[4].bias.data
            idbn2.running_mean.data[self.out_planes:]=self.conv[4].running_mean
            idbn2.running_var.data[self.out_planes:]=self.conv[4].running_var
            idrelu2 = nn.PReLU(self.mid_planes)
            torch.nn.init.ones_(idrelu2.weight.data[:self.in_planes])
            torch.nn.init.zeros_(idrelu2.weight.data[self.in_planes:])

            idconv3 = nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=1, bias=False).eval()
            idbn3=nn.BatchNorm2d(self.out_planes).eval()

            nn.init.dirac_(idconv3.weight.data[:,:self.out_planes])
            idconv3.weight.data[:,self.out_planes:],bias=self.fuse(self.conv[6].weight,self.conv[7].running_mean,self.conv[7].running_var,self.conv[7].weight,self.conv[7].bias,self.conv[7].eps)
            bn_var_sqrt=torch.sqrt(self.running2.running_var + self.running2.eps)
            idbn3.weight.data=bn_var_sqrt
            idbn3.bias.data=self.running2.running_mean
            idbn3.running_mean.data=self.running2.running_mean+bias
            idbn3.running_var.data=self.running2.running_var

            self.use_res_connect=False
            self.running1 = None
            self.running2 = None
            self.conv=nn.Sequential(*[idconv1,idbn1,idrelu1,idconv2,idbn2,idrelu2,idconv3,idbn3])
            
    def fuse(self,conv_w, bn_rm, bn_rv,bn_w,bn_b, eps):
        bn_var_rsqrt = torch.rsqrt(bn_rv + eps)
        conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
        conv_b = bn_rm * bn_var_rsqrt * bn_w-bn_b
        return conv_w,conv_b


class RMobileNet(nn.Module):
    def __init__(self, setting, width_mult,input_channel,output_channel,last_channel, n_class=100):
        super(RMobileNet, self).__init__()
        self.features = [conv_bn(3, input_channel, 2 if n_class==1000 else 1)]
        self.features.append(nn.Sequential(
                # dw
                nn.Conv2d(input_channel, input_channel, 3, stride=1, padding=1, groups=input_channel, bias=False),
                nn.BatchNorm2d(input_channel),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output_channel),
            ))
        input_channel = output_channel
        for t, c, n, s in setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                self.features.append(InvertedResidual(input_channel, output_channel, s, expand_ratio=t))
                input_channel = output_channel
        self.features.append(conv_1x1_bn(input_channel, last_channel))
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x
    def rm(self):
        for m in self.features:
            if isinstance(m,InvertedResidual):
                m.deploy()
        return self
    
    def deploy(self):
        self.rm()
        features=[]
        for m in self.features.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.BatchNorm2d) or isinstance(m,nn.PReLU) or isinstance(m,nn.ReLU6):
                features.append(m)
        new_features=[]
        while len(features)>3:
            if isinstance(features[0],nn.Conv2d) and isinstance(features[1],nn.BatchNorm2d) and isinstance(features[2],nn.Conv2d) and isinstance(features[3],nn.BatchNorm2d):
                conv,bn = fuse_cbcb(features[0],features[1],features[2],features[3])
                new_features.append(conv)
                new_features.append(bn)
                features=features[4:]
            else:
                new_features.append(features.pop(0))
        new_features+=features
        self.features=nn.Sequential(*new_features)
        return self
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
def mobilenetv1_cifar(n_class=100,width_mult=1,t_free=8):
    input_channel = int(32 * width_mult)
    output_channel = int(64 * t_free * width_mult)
    last_channel = 1280
    setting =[[1/t_free,64,1,2],
            [1,64,1,1],
            
            [2,128,1,2],
            [1,128,1,1],
            
            [2,256,1,2],
            [1,256,5,1],
            
            [2,512,1,2],
            [1,512,1,1]
        ]
    return RMobileNet(setting, width_mult,input_channel,output_channel,last_channel,n_class)


def mobilenetv1_imagenet(n_class=1000,width_mult=1,t_free=1):
    input_channel = int(32 * width_mult)
    output_channel = int(64 * t_free * width_mult)
    last_channel = 1280
    tt=3
    setting =[[1/t_free,24*tt,1,2],
                [1, 24*tt, 1, 1],
                [2, 32*tt, 1, 2],
                [1, 32*tt, 2, 1],
                [2, 64*tt, 1, 2],
                [1, 64*tt, 3, 1],
                [2, 96*tt, 1, 1],
                [1, 96*tt, 2, 1],
                [2, 160*tt, 1, 2],
                [1, 160*tt, 2, 1],
                [1, 320*tt*t_free, 1, 1],
            ]
    return RMobileNet(setting, width_mult,input_channel,output_channel,last_channel,n_class)

def mobilenetv1_imagenet_1(n_class=1000,width_mult=1,t_free=8,tt=4):
    input_channel = int(32 * width_mult)
    output_channel = int(64 * t_free * width_mult)
    last_channel = 1280
    setting =[[1/t_free,24*tt,1,2],
                [1, 24*tt, 1, 1],
                [1, 32*tt, 1, 2],
                [1, 32*tt, 2, 1],
                [1, 64*tt, 1, 2],
                [1, 64*tt, 3, 1],
                [1, 96*tt, 1, 1],
                [1, 96*tt, 2, 1],
                [1, 160*tt, 1, 2],
                [1, 160*tt, 2, 1],
                [1, 320*tt*t_free, 1, 1],
            ]
    return RMobileNet(setting, width_mult,input_channel,output_channel,last_channel,n_class)

def mobilenetv2(n_class=1000,pretrained=True,width_mult=1):
    input_channel = int(32 * width_mult)
    output_channel = int(16 * width_mult)
    last_channel = 1280
    setting = [[6,24,1,2],
            [6, 24, 1, 1],
            [6, 32, 1, 2],
            [6, 32, 2, 1],
            [6, 64, 1, 2],
            [6, 64, 3, 1],
            [6, 96, 1, 1],
            [6, 96, 2, 1],
            [6, 160, 1, 2],
            [6, 160, 2, 1],
            [6, 320, 1, 1],]

    
    model = RMobileNet(setting, width_mult,input_channel,output_channel,last_channel,n_class)
    if pretrained:
        pretrain_model=torchvision.models.mobilenet_v2(pretrained=True)
        for i in range(1,18):
            blocks=[]
            for m in pretrain_model.features[i].modules():
                if isinstance(m,nn.Conv2d) or isinstance(m,nn.BatchNorm2d) or isinstance(m,nn.ReLU6):
                    blocks.append(m)
            pretrain_model.features[i].conv=nn.Sequential(*blocks)
        pretrain_model.features[1]=pretrain_model.features[1].conv
        print(model.load_state_dict(pretrain_model.state_dict(),strict=False))
    return model
