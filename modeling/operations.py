from turtle import forward
from numpy import identity, pad
import torch
import torch.nn as nn
import torch.nn.functional as F


OPS = {
    'skip': lambda C,stride : Identity(),
    'conv1': lambda C,stride : Conv(C,C,1,stride,0),
    'conv3': lambda C,stride : Conv(C,C,3,stride,1),
    'conv5': lambda C,stride : Conv(C,C,5,stride,2),
    'dconv3_2': lambda C,stride : DilConv(C,C,3,stride,2,2),
    'dconv3_4': lambda C,stride : DilConv(C,C,3,stride,4,4),
    'dconv3_8': lambda C,stride : DilConv(C,C,3,stride,8,8),
    'dconv5_2': lambda C,stride : DilConv(C,C,5,stride,4,2),
    'dconv5_4': lambda C,stride : DilConv(C,C,5,stride,8,4),
    'dconv5_8': lambda C,stride : DilConv(C,C,5,stride,16,8),
}

class Identity(nn.Module):
    def __init__(self):
        super(Identity,self).__init__()

    def forward(self,x):
        return x

class Conv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding,
                        inplace=False,affine=True):
        super(Conv,self).__init__()
        self.op = nn.Sequential(
            nn.LeakyReLU(inplace=inplace),
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size,
                    stride=stride,
                    padding=padding),
        )

    def forward(self,x):
        return self.op(x)

class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation,
                        inplace=False,affine=True):
        super(DilConv,self).__init__()
        self.op = nn.Sequential(
            nn.LeakyReLU(inplace=inplace),
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation),
        )

    def forward(self,x):
        return self.op(x)

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
