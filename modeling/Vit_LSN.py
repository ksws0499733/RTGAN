import sys
from turtle import forward
# print(sys.path)
sys.path.append(r"E:\G\论文——铁路轨道识别\代码\2-RailSeg-tansformer")

import collections
import torch
import torch.nn as nn
import torch.nn.functional as F


from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.backbone import build_backbone

class Vit_LSN(nn.Module):
    def __init__(self,backbone='vit',output_stride = 16 , num_classes=21, in_chans=3,
                 pretrain=True):
        super(Vit_LSN, self).__init__()

        self.backbone, self.backbone_para = build_backbone(backbone,
                                            in_chans=in_chans, 
                                            output_stride = output_stride, 
                                            pretrain= pretrain)

        self.neck = neckNet(C_hid=64, 
                            num_layers=5, 
                            up_layers=[0,1,2,3],
                            layer_chan=self.backbone_para['layer_chan']
                            )
        self.head = headNet(C_dim=64,num_class= num_classes)

    def forward(self, input):
        C = self.backbone(input)  
        #c1 64/4
        #c2 64/8
        #c3 768/16
        #c4 768/16
        P = self.neck(C)
        #p1 64/8
        #p2 64/8
        #p3 64/4
        #p4 64/2    
        out_cls, out_side = self.head(P)
        return  {'cls':out_cls, 'side':out_side}

from .operations import Cell
class neckNet(nn.Module):
    def __init__(self, C_hid, num_layers, up_layers, layer_chan):
        super(neckNet, self).__init__()
        C = C_hid
        geno2=[
            [('conv5',0),('skip',0),('dconv5_8',1)],
            [('conv5',0),('skip',0),('dconv5_8',1)],
            [('dconv3_2',0),('dconv5_4',0),('conv3',1),('conv5',3)],
            [('dconv5_2',0),('dconv5_2',0),('conv1',1),('dconv3_2',2)],
            [('skip',0),('skip',1),('skip',0),('skip',2)],
            [('skip',0),('skip',1),('skip',0),('skip',2)],
        ]
        self.dsn = nn.ModuleList(
            [nn.Conv2d(Ci, C, 1) for Ci in layer_chan]
        )

        self.cat2 = nn.ModuleList(
            [nn.Conv2d(C*2, C, 1) for i in range(15)]
        )
        self.cat3 = nn.ModuleList(
            [nn.Conv2d(C*3, C, 1) for i in range(15)]
        )

        self.super = nn.ModuleList(
            [nn.Conv2d(C, 1, 1) for i in range(5)]
        )        
        
        self.ucells = nn.ModuleList(
            [Cell(geno2[i],C,C, 1 if i in up_layers else 0) for i in range(num_layers)]
        ) 
        self.up = lambda x,size : F.interpolate(x, size=size, mode='bilinear', align_corners=False)


    def forward(self, input):
        c1,c2,c3,c4 = input

        d4 = self.dsn[0](c1)# 1/4
        d3 = self.dsn[1](c2)# 1/8
        d2 = self.dsn[2](c3)# 1/16
        d1 = self.dsn[3](c4)# 1/16

        p1 = self.ucells[0](d1) #1/8

        p2 = self.ucells[1](d2)#1/8

        d3_2 = F.leaky_relu(
            self.cat3[1](
                torch.cat([p1,p2,d3],dim=1)
            ),
            inplace=True
        )
        p3 = self.ucells[2](d3_2) #1/4

        d4_2 = F.leaky_relu(
            self.cat2[2](
                torch.cat([p3, self.up(d4, p3.size()[2:4])],dim=1)
            ),
            inplace=True
        )
        p4 = self.ucells[3](d4_2) #1/2

        return p1,p2,p3,p4


class headNet(nn.Module):
    def __init__(self, C_dim, num_class):
        super(headNet, self).__init__()
        C = C_dim
        geno3 = [('dconv3_2',0),('dconv5_4',1),('dconv5_8',2),('conv1',0)]

        self.super = nn.ModuleList(
            [nn.Conv2d(C, num_class, 1) for i in range(5)]
        )        

        self.fuse_cell = Cell(geno3,C,C, 1)
        self.classifier1 = nn.Conv2d(C,num_class,1)
        self.classifier2 = nn.Conv2d(num_class,num_class,kernel_size=15,stride=1, padding=7)
        self.up = lambda x,size : F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, input):
        p1,p2,p3,p4 = input# 1/8, 1/8, 1/4, 1/2
        b,c,h,w = p4.shape

        p1_2 = self.up(p1, size=p4.size()[2:])
        p2_2 = self.up(p2, size=p4.size()[2:])
        p3_2 = self.up(p3, size=p4.size()[2:])
        p4_2 = p4



        # p_fuse = p1+p2+p3+p4+p5
        p_fuse = self.fuse_cell(p1_2*0+p2_2*0+p3_2*1+p4_2*1) # 1/1
        p_out = self.classifier2(self.classifier1(p_fuse))

        out1 = self.up(self.super[0](p1), (h*2,w*2))
        out2 = self.up(self.super[1](p2), (h*2,w*2))
        out3 = self.up(self.super[2](p3), (h*2,w*2))
        out4 = self.up(self.super[3](p4), (h*2,w*2))

        return p_out, [out1,out2,out3,out4]





if __name__ == "__main__":
    model = Vit_lsn(output_stride=16,num_classes=2)
    model.eval()
    input = torch.rand(1, 3, 256, 256)
    output = model(input)
    print(output[0].shape)
  

