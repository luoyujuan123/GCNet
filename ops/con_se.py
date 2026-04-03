import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import pdb
class SEBlock_four(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock_four, self).__init__()
        
        # Squeeze: Global Average Pooling to get channel-wise statistics
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Output size is (N, C, 1, 1)

        # Excitation: Fully connected layers with a reduction to control model complexity
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size, channels, _, _ = x.size()  # Input shape is (N, C, H, W)

        # Squeeze: Global average pooling
        y = self.global_avg_pool(x).view(batch_size, channels)  # Shape (N, C)

        # Excitation: Fully connected layers
        y = self.fc1(y)  # Shape (N, C // reduction)
        y = self.relu(y)
        y = self.fc2(y)  # Shape (N, C)
        y = self.sigmoid(y)  # Shape (N, C)

        # Scale: Reshape and multiply the input tensor by the scaling factors
        y = y.view(batch_size, channels, 1, 1)  # Shape (N, C, 1, 1)
        return x * y  # Element-wise multiplication with original input

class conblock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 64,
            **kwargs,
    ):
        super().__init__()
        self.seblock=SEBlock_four(128)
        self.conv1b3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.conv1a3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.conv33 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1, bias=False,
                        groups=hidden_dim),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
        )
        # self.conv1b5 = nn.Sequential(
        #     nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1),
        #     nn.InstanceNorm2d(hidden_dim),
        #     nn.SiLU(),
        # )
        # self.conv1a5 = nn.Sequential(
        #     nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1),
        #     nn.InstanceNorm2d(hidden_dim),
        #     nn.SiLU(),
        # )
        # self.conv55 = nn.Sequential(
        #     nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=5, stride=1, padding=2, bias=False,
        #                 groups=hidden_dim),
        #     nn.InstanceNorm2d(hidden_dim),
        #     nn.SiLU(),
        # )
        self.conv1b7 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1),
            nn.InstanceNorm2d(2*hidden_dim),
            nn.SiLU(),
        )
        self.conv1a7 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.conv77 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, stride=1, padding=3, bias=False,
                        groups=hidden_dim),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.finalconv11 = nn.Conv2d(in_channels=hidden_dim * 3, out_channels=hidden_dim, kernel_size=1, stride=1)
       
    def forward(self,x):
        # x1,x2=torch.chunk(x,chunks=2,dim=1)
        # x1=self.seblock(x1)
        # x1=self.seblock(x1)
        # output2=self.conv77(self.conv55(self.conv33(x2)))
        # # output1 = self.conv33(self.conv55(self.conv77(x1)))
        # # output2 = self.conv1b7(self.conv77(self.conv1b7(output2)))
        # #pdb.set_trace()
        # output = torch.cat((output1,output2),dim=1)
        #output = self.finalconv11(output)
        x1=self.seblock(x)
        x1=self.seblock(x1)
        x2=self.conv1a7(self.conv77(self.conv1b7(x)))
        x3=self.conv1a3(self.conv33(self.conv1b3(x)))
        output=torch.cat((x1, x2, x3), dim=1)
        output=self.finalconv11(output)
        return output




    


