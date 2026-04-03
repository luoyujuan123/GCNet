# Code for "TDN: Temporal Difference Networks for Efficient Action Recognition"
# arXiv: 2012.10071
# Limin Wang, Zhan Tong, Bin Ji, Gangshan Wu
# tongzhan@smail.nju.edu.cn

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.init import normal_, constant_
import torch.nn.functional as F
from ops.base_module import *
import pdb


class TDN_Net(nn.Module):

    #def __init__(self,resnet_model,resnet_model1,bash_model,apha,belta):
    def __init__(self,resnet_model,resnet_model1,bash_model,cb,apha,belta):
        super(TDN_Net, self).__init__()

        self.conv1 = list(resnet_model.children())[0]
        self.bn1 = list(resnet_model.children())[1]
        self.relu = nn.ReLU(inplace=True)

        self.bash_model=bash_model
        self.cb=cb
        #GTM
        self.conv1dfirst = nn.Conv1d(in_channels=128, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.convidsecond = nn.ConvTranspose1d(in_channels=2, out_channels=128, kernel_size=1, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
        #Mamba
        # self.conv1dfirst = nn.Conv1d(in_channels=256, out_channels=2, kernel_size=1, stride=1, padding=0)
        # self.convidsecond = nn.ConvTranspose1d(in_channels=2, out_channels=256, kernel_size=1, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
        
        self.norm=nn.LayerNorm(3136)        
        
        # implement conv1_5 and inflate weight 
        self.conv1_temp = list(resnet_model1.children())[0]
        params = [x.clone() for x in self.conv1_temp.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (3 * 4,) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        self.conv1_5 = nn.Sequential(nn.Conv2d(12,64,kernel_size=7,stride=2,padding=3,bias=False),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.conv1_5[0].weight.data = new_kernels

        self.maxpool_diff = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.resnext_layer1 =nn.Sequential(*list(resnet_model1.children())[4])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1_bak = nn.Sequential(*list(resnet_model.children())[4])
        self.layer2_bak = nn.Sequential(*list(resnet_model.children())[5])
        self.layer3_bak = nn.Sequential(*list(resnet_model.children())[6])
        self.layer4_bak = nn.Sequential(*list(resnet_model.children())[7])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avg_diff = nn.AvgPool2d(kernel_size=2,stride=2)
        self.fc = list(resnet_model.children())[8]
        self.apha = apha
        self.belta = belta

    def forward(self, x):
        x1, x2, x3, x4, x5 = x[:,0:3,:,:], x[:,3:6,:,:], x[:,6:9,:,:], x[:,9:12,:,:], x[:,12:15,:,:]
        x_c5 = self.conv1_5(self.avg_diff(torch.cat([x2-x1,x3-x2,x4-x3,x5-x4],1).view(-1,12,x2.size()[2],x2.size()[3])))
        x_diff = self.maxpool_diff(1.0/1.0*x_c5)
        
        temp_out_diff1 = x_diff 
        x_diff = self.resnext_layer1(x_diff)

        x = self.conv1(x3)
        x = self.bn1(x)
        x = self.relu(x)
        #fusion layer1
        x = self.maxpool(x)
        temp_out_diff1 = F.interpolate(temp_out_diff1, x.size()[2:])
        x = self.apha*x + self.belta*temp_out_diff1
        #fusion layer2
        x = self.layer1_bak(x)
        x_diff = F.interpolate(x_diff, x.size()[2:])
        x = self.apha*x + self.belta*x_diff


        B,C,H,W=x.shape#bs,c,h,w
        con_x=x
        rx=x
        rx,con_x=torch.chunk(con_x,chunks=2,dim=1)
        conv_x=self.cb(con_x)
        


        rx=rx.flatten(2)#bs,c,hw
        rx=self.conv1dfirst(rx)#64,1,56*56
        rx=self.norm(rx)
        rx=rx.view(rx.size()[0:2]+(-1,56))#bs,1,h,w
        rx = rx.view((-1, 8) + rx.size()[1:]) #b,s,c,h,w
        rx=rx.view(rx.size()[0:2]+(-1,))#b,s,chw
        rx=self.bash_model(rx)
        # rx = rx.transpose(1,2).contiguous()

        rx=rx.view((-1,)+rx.size()[2:])#bs,chw
        rx=rx.view(rx.size()[0:1]+(-1,3136))#bs,c,hw
        # rx=self.block(rx,H,W)
        # rx = rx.transpose(1,2).contiguous()
        rx=self.convidsecond(rx)

        rx=rx.view(rx.size()[0:2]+(-1,56))#bs,c,h,w

        x_out=torch.cat((rx,conv_x),dim=1)

        x=0.5*x_out+0.5*x
        
        x = self.layer2_bak(x)
        x = self.layer3_bak(x)
        x = self.layer4_bak(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        #pdb.set_trace()

        return x

def tdn_net(base_model=None,num_segments=8,bash_model=None,cb=None,pretrained=True, **kwargs):
#def tdn_net(base_model=None,num_segments=8,bash_model=None,pretrained=True, **kwargs):
    if("50" in base_model):
        resnet_model = fbresnet50(num_segments, pretrained)
        resnet_model1 = fbresnet50(num_segments, pretrained)
    else:
        resnet_model = fbresnet101(num_segments, pretrained)
        resnet_model1 = fbresnet101(num_segments, pretrained)

    if(num_segments == 8):
        model = TDN_Net(resnet_model,resnet_model1,apha=0.5,belta=0.5,bash_model=bash_model,cb=cb)
        #model = TDN_Net(resnet_model,resnet_model1,apha=0.5,belta=0.5,bash_model=bash_model)
    else:
        model = TDN_Net(resnet_model,resnet_model1,apha=0.75,belta=0.25,bash_model=bash_model,cb=cb)
        #model = TDN_Net(resnet_model,resnet_model1,apha=0.75,belta=0.25,bash_model=bash_model)
    return model