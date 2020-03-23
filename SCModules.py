import torch.nn as nn
import torch.nn.functional as F

class SCModule(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(SCModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pad = nn.ReflectionPad2d(2)
        
        self.conv = nn.Conv2d(in_channels, out_channels, 5, stride=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        
        x1 = self.conv(self.pad(x1))
        x2 = self.conv(self.pad(x2))
        x3 = self.conv(self.pad(x3))

        o1 = self.bn1(x1)
        o2 = self.bn2(x2)
        o3 = self.bn3(x3)

        #print(torch.cat([o_1_1,o_1_2,o_1_3],dim=1).size())

        return [o1,o2,o3]

class ResPoolModule(nn.Module):
    def __init__(self,in_channels):
        super(ResPoolModule, self).__init__()
        self.in_channels = in_channels

        self.bigpool = nn.MaxPool2d(4,stride=4)
        self.midpool = nn.MaxPool2d(2,stride=2)
        #self.smlpool = nn.MaxPool2d(2,stride=1)
        
        #self.mrgpool = nn.MaxPool1d(3)
        
        self.bn4 = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        x_1_1 = self.bigpool(x[0]) 
        x_1_2 = self.midpool(x[1]) 
        x_1_3 = x[2]#self.smlpool(x[2]) 
        
        x_1 = torch.stack([x_1_1,x_1_2,x_1_3])
        
        x_1 = torch.max(x_1,0)[0]
        
        return self.bn4(x_1)

class SCBottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,downsample = False):
        super(SCBottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        bottleneck_channel = out_channels//4
        self.downsample = downsample

        self.pad = nn.ReflectionPad2d(1)
        
        stride = 1
        
        if downsample and not in_channels == 64:
            stride=2
        
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channel, 1, stride=1, bias=False)
        self.conv2 = SCModule(bottleneck_channel, bottleneck_channel)
        self.conv3 = nn.Conv2d(bottleneck_channel, out_channels, 1, stride=1, bias=False)
        
        self.relu = nn.ReLU(inplace = True)
        
        self.bn1_1 = nn.BatchNorm2d(bottleneck_channel)
        self.bn1_2 = nn.BatchNorm2d(bottleneck_channel)
        self.bn1_3 = nn.BatchNorm2d(bottleneck_channel)
        
        self.bn3_1 = nn.BatchNorm2d(out_channels)
        self.bn3_2 = nn.BatchNorm2d(out_channels)
        self.bn3_3 = nn.BatchNorm2d(out_channels)
        
        self.downsampler = None
        if downsample:
            self.downsampler = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,stride=1,bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
    def forward(self, x):
        identity1 = x[0]
        identity2 = x[1]
        identity3 = x[2]
        out1 = x[0]
        out2 = x[1]
        out3 = x[2]
        
        # BIG
        out = self.conv1(out1)
        out = self.bn1_1(out)
        out1 = self.relu(out)
        
        out = self.conv1(out2)
        out = self.bn1_2(out)
        out2 = self.relu(out)
        
        out = self.conv1(out3)
        out = self.bn1_3(out)
        out3 = self.relu(out)
        
        out = self.conv2([out1,out2,out3]) # SCBlock
        
        # BIG
        out1 = self.conv3(out[0])
        out1 = self.bn3_1(out1)
        
        if self.downsample == True:
            identity1 = self.downsampler(identity1)
            
        out1 += identity1
        out1 = self.relu(out1)
        
        # MIDDLE
        
        out2 = self.conv3(out[1])
        out2 = self.bn3_2(out2)
        
        if self.downsample == True:
            identity2 = self.downsampler(identity2)
            
        out2 += identity2
        out2 = self.relu(out2)
        
        #SMALL
        
        out3 = self.conv3(out[2])
        out3 = self.bn3_3(out3)
        
        if self.downsample == True:
            identity3 = self.downsampler(identity3)
            
        out3 += identity3
        out3 = self.relu(out3)
        
        return [out1,out2,out3]