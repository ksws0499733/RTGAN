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


class Csp_LSN(nn.Module):
    def __init__(self, backbone=None, output_stride = 16 , num_classes=21, in_chans=3,
                 pretrain=True):
        super(Csp_LSN, self).__init__()
        print('\t----[Csp_LSN]----__init__')
        backbone = backbone or 'cspdarknet53'
        self.backbone, self.backbone_para = build_backbone(backbone,
                                                            in_chans=in_chans, 
                                                            output_stride = output_stride, 
                                                            out_indices=[0,1,2,3,5],
                                                            pretrain= pretrain)
        self.neck = neckNet(C_hid=64, 
                            num_layers=5, 
                            up_layers=[0,1,2,3],
                            layer_chan=self.backbone_para['layer_chan'])
        self.head = headNet(C_dim=64,num_class= num_classes)


        # self.grad = gradNet()

    def forward(self, input):
        c1,c2,c3,c4,c5 = self.backbone(input)
        #c1 64/1
        #c2 128/2
        #c3 256/4
        #c4 512/8
        #c5 2048/16
        p1,p2,p3,p4,p5 = self.neck([c1,c2,c3,c4,c5])
        #p1 64/8
        #p2 64/4
        #p3 64/2
        #p4 64/1
        #p5 64/1        

        out_cls, out_side = self.head([p1,p2,p3,p4,p5])

        if torch.isnan(input).any():
            print('input nan')
        for isd, c in enumerate([c1,c2,c3,c4,c5]):
            if torch.isnan(c).any():
                print('c %s nan'%isd)

        for isd, p in enumerate([p1,p2,p3,p4,p5]):
            if torch.isnan(p).any():
                print('p %s nan'%isd)

        if torch.isnan(out_cls).any():
            print('out_cls nan')

        return  {'cls':out_cls, 
                    'side':out_side, 
                    'bk_side':[c1,c2,c3,c4,c5],
                    'nk_side':[p1,p2,p3,p4,p5]}
        # out_dirct, _ = self.grad(out_cls[:,1:])

        # return  {'cls':out_cls, 'side':out_side, 'grad':out_dirct}


from .operations import Cell

class neckNet(nn.Module):
    def __init__(self, C_hid, num_layers, up_layers, layer_chan):
        super(neckNet, self).__init__()
        C = C_hid
        geno2=[
            [('dconv5_8',0),('skip',1),('conv5',0),('conv5',0)],
            [('dconv3_8',0),('dconv3_4',0),('skip',1),('conv5',0)],
            [('conv3',0),('skip',1),('conv5',2),('dconv5_2',2)],
            [('skip',0),('skip',1),('skip',0),('skip',2)],
            [('skip',0),('skip',1),('skip',0),('skip',2)],
            [('skip',0),('skip',1),('skip',0),('skip',2)],
        ]
        self.dsn = nn.ModuleList(
            [nn.Conv2d(Ci, C, 1) for Ci in layer_chan]
        )

        self.cat = nn.ModuleList(
            [nn.Conv2d(C*2, C, 1) for i in range(15)]
        )

        self.super = nn.ModuleList(
            [nn.Conv2d(C, 1, 1) for i in range(5)]
        )        
        
        self.ucells = nn.ModuleList(
            [Cell(geno2[i],C,C, 1 if i in up_layers else 0) for i in range(num_layers)]
        ) 
        self.up = lambda x,size : F.interpolate(x, size=size, mode='bilinear', align_corners=False)


    def forward(self, input):
        c1,c2,c3,c4,c5 = input

        d5 = self.dsn[0](c1)# 1/1
        d4 = self.dsn[1](c2)# 1/2
        d3 = self.dsn[2](c3)# 1/4
        d2 = self.dsn[3](c4)# 1/8
        d1 = self.dsn[4](c5)# 1/16

        p1 = self.up(self.ucells[0](d1), d2.size()[2:4]) #1/8

        d2_u = self.up(d2,p1.size()[2:])

        d2_2 = F.relu(
            self.cat[0](
                torch.cat(
                    [p1, d2_u],
                    dim=1
                )
            ),
            inplace=True
        )
        p2 = self.up(self.ucells[1](d2_2), d3.size()[2:4])#1/4

        d3_u = F.interpolate(d3, 
                            size=p2.size()[2:], 
                            mode='bilinear', 
                            align_corners=False)
        d3_2 = F.relu(
            self.cat[1](
                torch.cat([p2, d3_u],dim=1)
            ),
            inplace=True
        )
        p3 = self.up(self.ucells[2](d3_2), d4.size()[2:4]) #1/2

        p4 = self.up(self.ucells[3](d4), d5.size()[2:4]) #1
                        
        p5 = self.ucells[4](d5) #1   01

        return p1,p2,p3,p4,p5

    def crop(self,d,region):
        y,x,h,w = region
        d1 = d[:,:,y:y+h,x:x+w]
        return d1

class headNet(nn.Module):
    def __init__(self, C_dim, num_class):
        super(headNet, self).__init__()
        C = C_dim
        geno3 = [('dconv3_2',0),('dconv5_4',1),('dconv5_8',2),('conv1',0)]

        self.super = nn.ModuleList(
            [nn.Conv2d(C, num_class, 1) for i in range(5)]
        )        

        self.conv1 = nn.Conv2d(C*5,C,1)

        self.fuse_cell = Cell(geno3,C,C, 1)
        self.classifier1 = nn.Conv2d(C,num_class,1)
        self.classifier2 = nn.Conv2d(num_class,num_class,kernel_size=15,stride=1, padding=7)
        self.up = lambda x,size : F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, input):
        p1,p2,p3,p4,p5 = input# 1/8, 1/4, 1/2, 1/1, 1/1
        b,c,h,w = p5.shape

        # p1_2 = self.up(p1, size=p3.size()[2:])
        p2_2 = self.up(p2, size=p3.size()[2:])
        p3_2 = p3
        # p4_2 = self.up(p4, size=p3.size()[2:])#F.max_pool2d(p4, 2, 2, ceil_mode=True)
        # p5_2 = self.up(p5, size=p3.size()[2:])#F.max_pool2d(p5, 2, 2, ceil_mode=True)

        # p_fuse = self.conv1(torch.cat(
        #             [p1_2, p2_2,p3_2,p4_2,p5_2],
        #             dim=1
        #         ))
        # p_fuse = self.fuse_cell(p_fuse) # 1/1

        # p_fuse = p1+p2+p3+p4+p5
        # p_fuse = self.fuse_cell(p1_2*0+p2_2*1+p3_2*1+p4_2*0+p5_2*0) # 1/1
        p_fuse = self.fuse_cell(p2_2*1+p3_2*1) # 1/1
        p_out = self.classifier2(self.classifier1(p_fuse))
        # p_out = self.crop(p_out,(34,34,h,w))

        p_out = self.up(p_out, (h,w))
        p_out = F.softmax(p_out, dim=1)

        out1 = self.up(self.super[0](p1), (h,w))
        out2 = self.up(self.super[1](p2), (h,w))
        out3 = self.up(self.super[2](p3), (h,w))
        out4 = self.up(self.super[3](p4), (h,w))
        out5 = self.up(self.super[4](p5), (h,w))

        return p_out, [out1,out2,out3,out4,out5]

    def crop(self,d,region):
        y,x,h,w = region
        d1 = d[:,:,y:y+h,x:x+w]
        return d1


if __name__ == "__main__":
    model = Csp_LSN(output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 256, 256)
    output = model(input)
    print(output[0].shape)
  

