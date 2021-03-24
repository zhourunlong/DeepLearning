import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random

class VGG16_ensemble(nn.Module):
    def __init__(self, ensemble, seed):
        super(VGG16_ensemble, self).__init__()
        self.ensemble = ensemble
        vgg = []
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        # height*width*channel
        # 96*96*3 -> 48*48*64
        vgg.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.BatchNorm2d(64))
        vgg.append(nn.ReLU())
        vgg.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.BatchNorm2d(64))
        vgg.append(nn.ReLU())
        vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 48*48*64 -> 24*24*128
        vgg.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.BatchNorm2d(128))
        vgg.append(nn.ReLU())
        vgg.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.BatchNorm2d(128))
        vgg.append(nn.ReLU())
        vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 24*24*128 -> 12*12*256
        vgg.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.BatchNorm2d(256))
        vgg.append(nn.ReLU())
        vgg.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.BatchNorm2d(256))
        vgg.append(nn.ReLU())
        vgg.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.BatchNorm2d(256))
        vgg.append(nn.ReLU())
        vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 12*12*256 -> 6*6*512
        vgg.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.BatchNorm2d(512))
        vgg.append(nn.ReLU())
        vgg.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.BatchNorm2d(512))
        vgg.append(nn.ReLU())
        vgg.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.BatchNorm2d(512))
        vgg.append(nn.ReLU())
        vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 6*6*512 -> 3*3*512
        vgg.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.BatchNorm2d(512))
        vgg.append(nn.ReLU())
        vgg.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.BatchNorm2d(512))
        vgg.append(nn.ReLU())
        vgg.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.BatchNorm2d(512))
        vgg.append(nn.ReLU())
        vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.main = nn.Sequential(*vgg)

        classfication = []

        # 3*3*512 -> 10
        classfication.append(nn.Linear(in_features=512 * 3 * 3, out_features=10))

        self.classfication = nn.Sequential(*classfication)

    def forward(self, x):
        img_Q = []
        for batch in range(x.shape[0]):
            for i in range(self.ensemble):
                pos = np.random.random_integers(0, 32, size = 2)
                img_Q.append(x[batch, :, pos[0] : pos[0] + 96, pos[1] : pos[1] + 96])
        img = torch.cat(img_Q, dim=0).reshape(-1, 3, 96, 96)
        
        feature = self.main(img)
        feature = feature.view(img.size(0), -1)
        tmp = nn.Softmax(dim=1)(self.classfication(feature))

        result_Q = []
        for batch in range(x.shape[0]):
            i = batch * self.ensemble
            t = torch.sum(tmp[i : i + self.ensemble,:], dim=0)
            result_Q.append(t)
        result = torch.cat(result_Q, dim=0).reshape(-1, 10)

        result = torch.log(result / self.ensemble + 1e-8)
        return result
