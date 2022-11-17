import sys
# print(sys.path)
sys.path.append(r"E:\G\论文——铁路轨道识别\代码\2-RailSeg-tansformer")

import collections
import torch
import torch.nn as nn
import torch.nn.functional as F


from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.backbone import build_backbone
from .neck.aspp import ASPP
from .head.decoder import Decoder

class Deeplab(nn.Module):
    def __init__(self, backbone=None, output_stride = 16 , num_classes=21, in_chans=3,
                 pretrain=True):
        super(Deeplab, self).__init__()
        backbone = backbone or 'xception'
        self.backbone, self.backbone_para = build_backbone(backbone,
                                                            in_chans=in_chans, 
                                                            output_stride = output_stride)
        self.neck =  ASPP(output_stride=output_stride, 
                            inplanes = self.backbone_para['layer_chan'][0],
                            BatchNorm = nn.BatchNorm2d)

        self.head = Decoder( num_classes=num_classes, 
                            inplanes=[256,self.backbone_para['layer_chan'][1]], 
                            BatchNorm= nn.BatchNorm2d)
        self.up = lambda x,size : F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        
    def forward(self, input):
        c1,low_level = self.backbone(input)

        p5 = self.neck(c1)
 
        ocls, oside = self.head([p5,low_level])
        out_cls = self.up(ocls, input.size()[2:])

        out_side = []        
        for side in oside:
            side = self.up(side, input.size()[2:])
            out_side.append(side)

        return  {'cls':out_cls, 'side':out_side}

