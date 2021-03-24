
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
 
#bn层是放在conv层的前面和后面都可以，一般是放在后面，这里放在了前面
def conv_block(in_channel, out_channel):
    layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
    )
    return layer
 
 
#稠密块由多个conv_block 组成，每块使⽤用相同的输出通道数。但在前向计算时，我们将每块的输入和输出在通道维上连结。
class dense_block(nn.Module):
    # growth_rate即output_channel
    def __init__(self, in_channel, growth_rate, num_layers):
        super(dense_block, self).__init__()
        block = []
        channel = in_channel
        for i in range(num_layers):
            block.append(
                conv_block(in_channel=channel, out_channel=growth_rate)
            )
            channel += growth_rate
            self.net = nn.Sequential(*block)
 
    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x

def conv_block(in_channel, out_channel):
    layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
    )
    return layer
 
 
#稠密块由多个conv_block 组成，每块使⽤用相同的输出通道数。但在前向计算时，我们将每块的输入和输出在通道维上连结。
class dense_block(nn.Module):
    # growth_rate即output_channel
    def __init__(self, in_channel, growth_rate, num_layers):
        super(dense_block, self).__init__()
        block = []
        channel = in_channel
        for i in range(num_layers):
            block.append(
                conv_block(in_channel=channel, out_channel=growth_rate)
            )
            channel += growth_rate
            self.net = nn.Sequential(*block)
 
    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x

def transition_block(in_channel, out_channel):
    trans_layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel, out_channel, 1),
        nn.AvgPool2d(2, 2)
    )
    return trans_layer

def conv_block(in_channel, out_channel):
    layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
    )
    return layer
 
 
#稠密块由多个conv_block 组成，每块使⽤用相同的输出通道数。但在前向计算时，我们将每块的输入和输出在通道维上连结。
class dense_block(nn.Module):
    # growth_rate即output_channel
    def __init__(self, in_channel, growth_rate, num_layers):
        super(dense_block, self).__init__()
        block = []
        channel = in_channel
        for i in range(num_layers):
            block.append(
                conv_block(in_channel=channel, out_channel=growth_rate)
            )
            channel += growth_rate
            self.net = nn.Sequential(*block)
 
    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x

class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_layers=[6, 12, 36, 24]):
        super(DenseNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=3, padding=10),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
 
        channels = 64
        block = []
        for i, layers in enumerate(block_layers):
            block.append(dense_block(channels, growth_rate, layers))
            channels += layers * growth_rate
            if i != len(block_layers) - 1:
                block.append(transition_block(channels, channels // 2)) # 通过 transition 层将大小减半， 通道数减半
                channels = channels // 2
        self.block2 = nn.Sequential(*block)
        self.block2.add_module('bn', nn.BatchNorm2d(channels))
        self.block2.add_module('relu', nn.ReLU(True))
        self.block2.add_module('avg_pool', nn.AvgPool2d(3))
        self.classifier = nn.Linear(channels, 10)
 
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
