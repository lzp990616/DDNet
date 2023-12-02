import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

def expend_as(tensor, rep):
    my_repeat = tensor.repeat(1, 1, 1, rep)
    return my_repeat

class Channelblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Channelblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) # torch.Size([1, 128, 1, 1])
        
        self.dense1 = nn.Linear(out_channels*2, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.dense2 = nn.Linear(out_channels, out_channels)
        self.sigmoid = nn.Sigmoid()
        self.out_channels = out_channels
        
    def forward(self, x):
        # pdb.set_trace()
        conv1 = self.conv1(x)
        batch1 = self.bn1(conv1)
        leakyReLU1 = F.relu(batch1)
        
        conv2 = self.conv2(x)
        batch2 = self.bn2(conv2)
        leakyReLU2 = F.relu(batch2)
        
        data3 = torch.cat((leakyReLU1, leakyReLU2), dim=1)
        data3 = self.global_pool(data3)
        # data3 = F.avg_pool2d(data3, data3.size()[2:])
        data3 = data3.view(data3.size(0), -1)
        data3 = self.dense1(data3) 
        data3 = data3.unsqueeze(-1).unsqueeze(-1)	            
        data3 = self.bn3(data3)
        data3 = F.relu(data3)
        
        data3 = data3.view(data3.size(0), -1)
        data3 = self.dense2(data3) # 64,64
        data3 = self.sigmoid(data3)
        a = data3.unsqueeze(-1).unsqueeze(-1)
        # a = data3.view(x.size(0), 1, 1, self.out_channels)
        # a = data3.view(-1, self.bn1.weight.shape[0], 1, 1)
        
        a1 = 1 - data3
        a1 = a1.unsqueeze(-1).unsqueeze(-1)
        y = torch.mul(leakyReLU1, a)
        y1 = torch.mul(leakyReLU2, a1)
        data_a_a1 = torch.cat((y, y1), dim=1)
        conv3 = nn.Conv2d(self.out_channels*2, self.out_channels, kernel_size=1, padding=0)(data_a_a1)
        batch3 = nn.BatchNorm2d(self.out_channels)(conv3)
        leakyReLU3 = F.relu(batch3)
        
        return leakyReLU3

class SpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 1):
        super(SpatialBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv4 = nn.Conv2d(out_channels*2, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn3 = nn.BatchNorm2d(out_channels)
        # self.bn4 = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, channel_data):
        
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = F.relu(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        relu2 = F.relu(bn2)

        data3 = channel_data + relu2
        relu3 = F.relu(data3)
        # pdb.set_trace()
        sigmoid3 = self.sigmoid(self.conv3(relu3))
        
        a = sigmoid3.expand_as(relu2)
        y = a * channel_data
        
        a1 = 1 - sigmoid3
        a1 = a1.expand_as(relu2)
        y1 = a1 * relu2
        
        out = torch.cat([y, y1], dim=1)
        out = self.conv4(out)
        out = self.bn3(out)
        out = out + relu1
        
        return out

    
    
class DoubleConv(nn.Module):
    def __init__(self, in_channal, out_channal):
        super(DoubleConv, self).__init__()
        self.Channelblock = Channelblock(in_channal, out_channal)
        self.SpatialBlock = SpatialBlock(in_channal, out_channal)
    def forward(self, x):
        channel_data = self.Channelblock(x)
        x = self.SpatialBlock(x, channel_data)
        return x
