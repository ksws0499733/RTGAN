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

class Vit_Deeplab(nn.Module):
    def __init__(self, backbone=None, output_stride = 16 , num_classes=21, in_chans=3,
                 pretrain=True):
        super(Vit_Deeplab, self).__init__()

        self.backbone, self.backbone_para = build_backbone(backbone,
                                            in_chans=in_chans, 
                                            output_stride = output_stride, 
                                            pretrain= pretrain)

        self.neck =  ASPP(output_stride=output_stride, 
                            inplanes = self.backbone_para['layer_chan'][-1],
                            BatchNorm = nn.BatchNorm2d)

        self.head = Decoder( num_classes=num_classes, 
                            inplanes=[256,64], 
                            BatchNorm= nn.BatchNorm2d)

    def forward(self, input):
        c1,c2,c3,c4 = self.backbone(input)
        #c1 64/4
        #c2 64/8
        #c3 768/16
        #c4 768/16
        p4 = self.neck(c4)
        #p5 256/16       

        out_cls, out_side = self.head([p4,c1,input])

        return  {'cls':out_cls, 'side':out_side}
