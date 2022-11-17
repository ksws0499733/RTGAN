import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class Decoder(nn.Module):
    def __init__(self, num_classes, inplanes, BatchNorm):
        super(Decoder, self).__init__()
        input_dim,low_level_inplanes = inplanes        

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(input_dim+48, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        
        self.classifier1 = nn.Conv2d(256,num_classes,1)
        self.classifier2 = nn.Conv2d(num_classes,
                                    num_classes,
                                    kernel_size=15,
                                    stride=1, 
                                    padding=7)
        
        
        self.super = nn.ModuleList(
            [nn.Conv2d(C, num_classes, 1) for C in [input_dim,48]]
        )        
        self.up = lambda x,size : F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self._init_weight()


    def forward(self, input):
        p1,p3 = input# 1/16, 1/4

        p3 = self.conv1(p3)
        p3 = self.bn1(p3)
        p3 = self.relu(p3)

        p1 = self.up(p1, p3.size()[2:])
        p_fuse = torch.cat((p1, p3), dim=1)     
        p_fuse = self.last_conv(p_fuse)

        out = self.classifier2(p_fuse)

        out1 = self.super[0](p1)
        out3 = self.super[1](p3)

        return out, [out1,out3]


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm,attention):
    return Decoder(num_classes, backbone, BatchNorm,attention)