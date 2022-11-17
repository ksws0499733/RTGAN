import sys
# print(sys.path)
sys.path.append(r"E:\G\论文——铁路轨道识别\代码\2-RailSeg-tansformer")

import collections
import torch
import torch.nn as nn
import torch.nn.functional as F


from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.backbone import build_backbone
from .head.grad_head import gradNet
from mmcv.cnn import ConvModule,build_norm_layer

class SETR_MLA(nn.Module):
    def __init__(self, backbone=None, output_stride = 16 , 
                num_classes=21, 
                in_chans=3,
                 pretrain=True):
        super(SETR_MLA, self).__init__()
        print('\t----[SegFormer]----__init__')
        backbone = backbone or 'SETR'
        self.backbone, self.backbone_para = build_backbone(backbone,
                                            in_chans=in_chans, 
                                            output_stride = output_stride, 
                                            pretrain= pretrain)
        self.head = VIT_MLAHead(num_class=num_classes)

        self.up = lambda x,size : F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        out_feature = self.backbone(input)  

        out_cls = self.head(out_feature)

        out_cls =  self.softmax(self.up(out_cls, input.size()[2:]))


        return  {'cls':out_cls, 
                 'side':[]}


class MLAHead(nn.Module):
    def __init__(self, mla_channels=256, mlahead_channels=128, norm_cfg=None):
        super(MLAHead, self).__init__()
        self.head2 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                        build_norm_layer(norm_cfg, mlahead_channels)[1], 
                        nn.ReLU(),
                        nn.Conv2d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                        build_norm_layer(norm_cfg, mlahead_channels)[1], 
                        nn.ReLU())
        self.head3 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[
            1], nn.ReLU(),
            nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())
        self.head4 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[
            1], nn.ReLU(),
            nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())
        self.head5 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[
            1], nn.ReLU(),
            nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())

    def forward(self, mla_p2, mla_p3, mla_p4, mla_p5):
        # head2 = self.head2(mla_p2)
        head2 = F.interpolate(self.head2(
            mla_p2), 4*mla_p2.shape[-1], mode='bilinear', align_corners=True)
        head3 = F.interpolate(self.head3(
            mla_p3), 4*mla_p3.shape[-1], mode='bilinear', align_corners=True)
        head4 = F.interpolate(self.head4(
            mla_p4), 4*mla_p4.shape[-1], mode='bilinear', align_corners=True)
        head5 = F.interpolate(self.head5(
            mla_p5), 4*mla_p5.shape[-1], mode='bilinear', align_corners=True)
        return torch.cat([head2, head3, head4, head5], dim=1)


class VIT_MLAHead(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, 
                    img_size=512,
                    mla_channels=256, 
                    mlahead_channels=128,
                    norm_layer=nn.BatchNorm2d, 
                    num_class=3,
                    norm_cfg=dict(type='BN', requires_grad=True)):
        super(VIT_MLAHead, self).__init__()
        self.img_size = img_size
        self.norm_cfg = norm_cfg
        self.mla_channels = mla_channels
        self.BatchNorm = norm_layer
        self.mlahead_channels = mlahead_channels
        self.num_classes = num_class

        self.mlahead = MLAHead(mla_channels=self.mla_channels,
                               mlahead_channels=self.mlahead_channels, 
                               norm_cfg=self.norm_cfg)
        self.cls = nn.Conv2d(4 * self.mlahead_channels,
                             self.num_classes, 3, padding=1)

    def forward(self, inputs):
        x = self.mlahead(inputs[0], inputs[1], inputs[2], inputs[3])
        x = self.cls(x)
        # x = F.interpolate(x, size=self.img_size, mode='bilinear',
        #                   align_corners=False)
        return x

  

