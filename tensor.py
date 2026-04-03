import torch
import torch.nn as nn
x = torch.randn(32, 15, 224, 224)#bs,c,h,w
x=x.view((-1,3)+x.size()[2:])
print(x.size())
# x_in=x
# print(x.size())
# x=x.view((-1,16)+x.size()[1:])#b,s,c,h,w
# print(x.size())
# x=x.view(x.size()[0:2]+(-1,))
# print(x.size())
# x=x.transpose(1,2)
# print(x.size())
# x=x.transpose(1,2)
# print(x.size())
# print("1")
# x=x.view((-1,)+x.size()[2:])
# print(x.size())
# x=x.view(x.size()[0:1]+(-1,3136))
# print(x.size())
# x_out=x.view(x.size()[0:2]+(-1,56))
# print(x_out.size())



# conv1dfirst = nn.Conv1d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0)
# convidsecond = nn.ConvTranspose1d(in_channels=64, out_channels=256, kernel_size=1, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
# x=torch.randn(64,256,3136)
# print(x.size())
# x=conv1dfirst(x)
# print(x.size())
# x=convidsecond(x)
# print(x.size())

# x = torch.randn(64, 256, 56, 56)#bs,c,h,w
# dwconv = nn.Conv2d(256, 256, 3, 1, 1,bias=True, groups=256)
# x=dwconv(x)
# print(x.size())