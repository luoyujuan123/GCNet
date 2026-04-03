import torch
from thop import profile
from archs.ViT_model import get_vit, ViT_Aes
from torchvision.models import resnet50
from ops.models import TSN

model =  TSN(51,
                8,
                'RGB',
                'resnet50')  #自己定义的模型，但要保证前面保存的层和自定义的模型中的层一致 
input1 = torch.randn(4, 3, 224, 224) 
flops, params = profile(model, inputs=(input1, ))
print('FLOPs = ' + str(flops/1000**3) + 'G')
print('Params = ' + str(params/1000**2) + 'M')
