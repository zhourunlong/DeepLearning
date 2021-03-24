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
    def __init__(self):
        super(ResNet,self).__init__()
        self.conv_1 = nn.Sequential(
        nn.AvgPool2d(4), #128->32
        nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(64))
        #self.block1 = ResBlock(16,32,2) #128->64
        #self.block2 = ResBlock(32,64,2) #64->32
        #self.block1 = ResBlock(32,64,4) #128->32
        self.block3 = ResBlock(64,128,2) #32->16
        self.block4 = ResBlock(128,256,2) #16->8
        self.block5 = ResBlock(256,512,1)
        self.block6 = ResBlock(512,512,1)
        self.outlayer = nn.Linear(512,10)
        
    def forward(self, x):
        x = F.relu(self.conv_1(x))
        #x = self.block1(x) 
        #x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = F.adaptive_avg_pool2d(x,[1,1])
        x = x.reshape(x.size(0),-1)
        result = self.outlayer(x)
        return result   

