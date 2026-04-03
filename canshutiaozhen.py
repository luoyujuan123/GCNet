import torch
from collections import OrderedDict
import os
import torch.nn as nn
import torch.nn.init as init
from ops.models_mamba import TSN


def init_weight(modules):    
    for m in modules:        
        if isinstance(m, nn.Conv2d):            
            init.xavier_uniform_(m.weight.data)            
            if m.bias is not None:                
                m.bias.data.zero_()        
        elif isinstance(m, nn.BatchNorm2d):            
            m.weight.data.fill_(1)            
            m.bias.data.zero_()        
        elif isinstance(m, nn.Linear):            
            m.weight.data.normal(0,0.01)            
            m.bias.data.zero_() 
            
def copyStateDict(state_dict):    
    if list(state_dict.keys())[0].startswith('module'):        
        start_idx = 1    
    else:        
        start_idx = 0    
    new_state_dict = OrderedDict()    
    for k,v in state_dict.items():        
        name = ','.join(k.split('.')[start_idx:])        
        new_state_dict[name] = v    
    return new_state_dict 

#加载pretrain model
checkpoint = torch.load('/media/sdc/liujiayu/TDN-main/checkpoint/news/best.pth.tar') 

# new_dict = copyStateDict(state_dict)
keys = []
for k,v in checkpoint['state_dict'].items(): 
    #print(k)
    # if k.startswith('module.cb'):
    #     print(k)    #将‘conv_cls’开头的key过滤掉，这里是要去除的层的key        
    #     continue    
    # if k.startswith('module.bash_model.seblock'):
    #     print(k)    #将‘conv_cls’开头的key过滤掉，这里是要去除的层的key        
    #     continue    
    if k.startswith('module.base_model.conv1dfirst.weight'):
        print(k+'\n')    #将‘conv_cls’开头的key过滤掉，这里是要去除的层的key 
        print(v.size())
      
        #continue  
    # if k.startswith('base_model.bash_model.seblock'):    #将‘conv_cls’开头的key过滤掉，这里是要去除的层的key        
    #     continue   
    keys.append(k) 

#去除指定层后的模型
checkpoint['state_dict'] = {k:checkpoint['state_dict'][k] for k in keys} 

net = TSN(51,
                8,
                'RGB',
                'resnet50')  #自己定义的模型，但要保证前面保存的层和自定义的模型中的层一致 

#加载pretrain model中的参数到新的模型中，此时自定义的层中是没有参数的，在使用的时候需要init_weight一下
net.state_dict().update(checkpoint['state_dict']) 

#保存去除指定层后的模型
torch.save(checkpoint,'/media/sdc/liujiayu/TDN-main/checkpoint/news/best2.pth.tar')

# model.load_state_dict(torch.load('/media/hd0/liujiayu/code/TDN-main/checkpoint/HUMBD/best1.pth.tar'),strict=False)
