import torch
import torch.nn as nn
import torch.nn.functional as F

class SCModule(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(SCModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pad = nn.ReflectionPad2d(2)
        
        self.conv = nn.Conv2d(in_channels, out_channels, 5, stride=1, bias=False)
        
        self.relu = nn.ReLU(inplace = True)
        
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

        o1 = self.relu(self.bn1(x1))
        o2 = self.relu(self.bn2(x2))
        o3 = self.relu(self.bn3(x3))

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
    
# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        
        self.downsample = downsample
        
        stride = 1
        
        if downsample:
            stride = 2
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsampler = None
        if downsample:
            self.downsampler = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsampler(x)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,downsample = False):
        super(Bottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        bottleneck_channel = out_channels//4
        self.downsample = downsample

        self.pad = nn.ReflectionPad2d(1)
        
        stride = 1
        
        if downsample and not in_channels == 64 and not in_channels == 16:
            stride=2
        
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channel, 1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(bottleneck_channel, bottleneck_channel, 3, stride=stride, bias=False)
        self.conv3 = nn.Conv2d(bottleneck_channel, out_channels, 1, stride=1, bias=False)
        
        self.relu = nn.ReLU(inplace = True)
        
        self.bn1 = nn.BatchNorm2d(bottleneck_channel)
        self.bn2 = nn.BatchNorm2d(bottleneck_channel)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.downsampler = None
        if downsample:
            self.downsampler = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.pad(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample == True:
            identity = self.downsampler(x)
            
        out += identity
        out = self.relu(out)
        
        return out

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

class InputDivider(nn.Module):
    def __init__(self):
        super(InputDivider, self).__init__()

        self.prepool1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.prepool2 = nn.AvgPool2d(2,2)
        
    def forward(self, x):
        return [self.prepool1(x),x,self.prepool2(x)]
    
class SCBasicRes(nn.Module):
    def __init__(self,in_channels,out_channels,downsample = False):
        super(SCBasicRes, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample

        self.pad = nn.ReflectionPad2d(1)
        
        stride = 1
        
        self.conv1 = SCModule(in_channels, out_channels)
        self.conv2 = SCModule(out_channels, out_channels)
        
        self.relu = nn.ReLU(inplace = True)
        
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
        
        out = self.conv1([out1,out2,out3]) # SCBlock
        out = self.conv2(out) # SCBlcok
        
        # BIG
        
        if self.downsample == True:
            identity1 = self.downsampler(identity1)
            
        out[0] += identity1
        out[0] = self.relu(out[0])
        
        # MIDDLE
        
        if self.downsample == True:
            identity2 = self.downsampler(identity2)
            
        out[1] += identity2
        out[1] = self.relu(out[1])
        
        #SMALL
        
        if self.downsample == True:
            identity3 = self.downsampler(identity3)
            
        out[2] += identity3
        out[2] = self.relu(out[2])
        
        return out

class ResNet_or_SC(nn.Module):
    def __init__(self,SCblock,num_blocks,num_classes=10):
        super(ResNet_or_SC, self).__init__()
        
        block = []
        for sc in SCblock:
            if sc == True:
                block.append(SCBottleneck)
            else:
                block.append(Bottleneck)
        
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        self.layer1 = self._make_layer(block[0],num_blocks[0],64,256,SC=True)
        self.layer2 = self._make_layer(block[1],num_blocks[1],256,512,SC=True)
        self.layer3 = self._make_layer(block[2],num_blocks[2],512,1024,SC=True)
        self.layer4 = self._make_layer(block[3],num_blocks[3],1024,2048,SC=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048,num_classes)
        
    def _make_layer(self,block,num_layer_blocks,in_channels,out_channels,SC=False):
        
        layers = []
        for i in range(num_layer_blocks):
            if i == 0:
                if SC:
                    layers.append(InputDivider())
                layers.append(block(in_channels,out_channels,downsample=True))
            else:
                layers.append(block(out_channels,out_channels))
        if SC:
            layers.append(ResPoolModule(out_channels))
            
        return nn.Sequential(*layers)
        

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        #print(x.size())
            
        x = self.layer1(x)
        #print(x.size())
        x = self.layer2(x)
        #print(x.size())
        x = self.layer3(x)
        #print(x.size())
        x = self.layer4(x)
        #print(x.size())
        
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        #print(x.size())
        
        return x
    
class ResNet_for_cifar(nn.Module):
    def __init__(self,SCblock,num_blocks):
        super(ResNet_for_cifar, self).__init__()
        
        block = []
        for sc in SCblock:
            if sc == True:
                block.append(SCBottleneck)
            else:
                block.append(Bottleneck)
        
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace = True)
        
        self.layer1 = self._make_layer(block[0],num_blocks[0],16,16,SC=SCblock[0],downsample=False)
        self.layer2 = self._make_layer(block[1],num_blocks[1],16,32,SC=SCblock[1])
        self.layer3 = self._make_layer(block[2],num_blocks[2],32,64,SC=SCblock[2])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64,10)
        
    def _make_layer(self,block,num_layer_blocks,in_channels,out_channels,SC=False,downsample=True):
        
        layers = []
        for i in range(num_layer_blocks):
            if i == 0:
                if SC:
                    layers.append(InputDivider())
                layers.append(block(in_channels,out_channels,downsample=downsample))
            else:
                layers.append(block(out_channels,out_channels))
        if SC:
            layers.append(ResPoolModule(out_channels))
            
        return nn.Sequential(*layers)
        

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        #print(x.size())
            
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        
        return x
        
def SCResNet50(num_classes=10):
    return ResNet_or_SC([True,True,True,True],[3,4,6,3],num_classes)

def ResNet_cifar():
    return ResNet_for_cifar([False,False,False],[2,2,2])

def SCResNet_cifar():
    return ResNet_for_cifar([True,True,True],[2,2,2])