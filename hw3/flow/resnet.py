import torch
from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(self,ch_in,ch_out,stride):
        super(ResBlock,self).__init__()
        self.conv_1 = nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1)
        self.bn_1 = nn.BatchNorm2d(ch_out)
        self.conv_2 = nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn_2 = nn.BatchNorm2d(ch_out)
        self.ch_in,self.ch_out,self.stride = ch_in,ch_out,stride
        self.ch_trans = nn.Sequential()
        if ch_in != ch_out:
            self.ch_trans = nn.Sequential(nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride),nn.BatchNorm2d(self.ch_out))
        #ch_trans表示通道数转变。因为要做short_cut,所以x_pro和x_ch的size应该完全一致
        
    def  forward(self,x):
        x_pro = F.relu(self.bn_1(self.conv_1(x)))
        x_pro = self.bn_2(self.conv_2(x_pro))
        
        #short_cut:
        x_ch = self.ch_trans(x)
        out = x_pro + x_ch
        return out

class ResNet(nn.Module):
    def __init__(self, act):
        super(ResNet,self).__init__()
        self.conv_1 = nn.Sequential(
        nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(32))
        self.block3 = ResBlock(32,64,2) #28->14
        self.block4 = ResBlock(64,128,2) #14->7
        self.block5 = ResBlock(128,256,1)
        self.block6 = ResBlock(256,256,1)
        self.outlayer = nn.Linear(256,28*28)
        self.act = act
        
    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = F.adaptive_avg_pool2d(x,[1,1])
        x = x.reshape(x.size(0),-1)
        res = self.outlayer(x)
        return self.act(res)

