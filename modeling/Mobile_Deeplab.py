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

class Mobile_Deeplab(nn.Module):
    def __init__(self, backbone=None, output_stride = 16 , num_classes=21, in_chans=3,
                 pretrain=True):
        super(Mobile_Deeplab, self).__init__()
        backbone = backbone or 'mobilenetv3_large_100'
        self.backbone, self.backbone_para = build_backbone(backbone,
                                                            in_chans=in_chans, 
                                                            output_stride = output_stride, 
                                                            out_indices=[0,1,2,3,4],
                                                            pretrain= pretrain)
        self.neck =  ASPP(output_stride=output_stride, 
                            inplanes = self.backbone_para['layer_chan'][-1],
                            BatchNorm = nn.BatchNorm2d)

        self.head = Decoder( num_classes=num_classes, 
                            inplanes=(256, self.backbone_para['layer_chan'][1]), 
                            BatchNorm= nn.BatchNorm2d)

    def forward(self, input):
        c1,c2,c3,c4,c5 = self.backbone(input)

        #c1 64/2
        #c2 128/4
        #c3 256/8
        #c4 512/16
        #c5 2048/16
        p5 = self.neck(c5)
        #p5 256/16       

        out_cls, out_side = self.head([p5,c2,input])

        return  {'cls':out_cls, 'side':out_side}
