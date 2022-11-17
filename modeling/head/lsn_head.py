import torch
import torch.nn as nn
import torch.nn.functional as F
from ..operations import OPS

class Cell(nn.Module):
    def __init__(self, genotype,C_prev, C_curr, flag=0):
        super(Cell, self).__init__() 
        self.flag = flag
        self.C_prev = C_prev
        self.C_curr = C_curr
        self.up2 = lambda x : F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        op_names, indices = zip(*genotype)
        concat = range(0,len(op_names)+1)
        self._compile(C_prev, op_names, indices, concat)

    def _compile(self, C_prev, op_names, indices, concat):
        assert len(op_names)==len(indices)
        self._steps = len(op_names)
        self._concat = concat
        self._ops = nn.ModuleList()
        for name in op_names:
            op = OPS[name](C_prev, stride = 1)
            self._ops += [op]
        self._indices = indices
    
    def forward(self, s):
        states = [s]
        for i in range(self._steps):
            cur_s = states[self._indices[i]]
            cur_s = self._ops[i](cur_s)
            states += [cur_s]
        
        s = sum(states[i] for i in self._concat)
        if self.flag == 1:
            s = self.up2(s)
        return s

class neckNet(nn.Module):
    def __init__(self, C_hid, num_layers, up_layers):
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
            [nn.Conv2d(Ci, C, 1) for Ci in [64,64,256,256]]
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

    def crop(self,d,region):
        y,x,h,w = region
        d1 = d[:,:,y:y+h,x:x+w]
        return d1
