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
from mmcv.cnn import ConvModule

class SegFormer(nn.Module):
    def __init__(self, backbone=None, output_stride = 16 , 
                num_classes=21, 
                in_chans=3,
                 pretrain=True):
        super(SegFormer, self).__init__()
        print('\t----[SegFormer]----__init__')
        backbone = backbone or 'mit_b0'
        self.backbone, self.backbone_para = build_backbone(backbone,
                                                            in_chans=in_chans, 
                                                            output_stride = output_stride, 
                                                            out_indices=[0,1,2,3,5],
                                                            pretrain= pretrain)

        self.head = SegFormerHead(feature_strides=self.backbone_para['layer_stride'],
                                    in_channels = self.backbone_para['layer_chan'],
                                    num_classes = num_classes,
                                    embedding_dim=256)

        self.up = lambda x,size : F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        out_feature = self.backbone(input)  

        out_cls = self.head(out_feature)

        out_cls =  self.softmax(self.up(out_cls, input.size()[2:]))


        return  {'cls':out_cls, 
                 'side':[]}

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
        
class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides,
                in_channels,
                 num_classes,
                 embedding_dim=256,
                 dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = nn.Identity()
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels


        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        self.up = lambda x,size : F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):

        c1, c2, c3, c4 = inputs# len=4, 1/4,1/8,1/16,1/32

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = self.up(_c4, size=c1.size()[2:])

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = self.up(_c3, size=c1.size()[2:])

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = self.up(_c2, size=c1.size()[2:])

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


if __name__ == "__main__":
    model = SegFormer(output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 256, 256)
    output = model(input)
    print(output[0].shape)
  

