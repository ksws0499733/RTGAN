import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.backbone import build_backbone

class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveMaxPool2d(pool_size),
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                )
            )
            
    def forward(self, x):
        out_puts = []
        for ppm in self:
            ppm_out = nn.functional.interpolate(ppm(x),size=(x.size(2), x.size(3)),mode='bilinear', align_corners=True)
            out_puts.append(ppm_out)
        return out_puts
 
    
class PSPHEAD(nn.Module):
    def __init__(self, in_channels, out_channels,pool_sizes = [1, 2, 3, 6],num_classes=31):
        super(PSPHEAD, self).__init__()
        self.pool_sizes = pool_sizes
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.psp_modules = PPM(self.pool_sizes, self.in_channels, self.out_channels)
        self.final = nn.Sequential(
            nn.Conv2d(self.in_channels + len(self.pool_sizes)*self.out_channels, self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )
        
    def forward(self, x):
        out = self.psp_modules(x)
        out.append(x)
        out = torch.cat(out, 1)
        out = self.final(out)
        return out
 
class Pspnet(nn.Module):
    def __init__(self, backbone=None, output_stride = 16 , num_classes=21, in_chans=3,
                 pretrain=True):
        super(Pspnet, self).__init__()
        self.num_classes = num_classes
        backbone = backbone or 'cspdarknet53'
        self.backbone, self.backbone_para = build_backbone(backbone,
                                                            in_chans=in_chans, 
                                                            output_stride = output_stride, 
                                                            out_indices=[0,1,2,3,5],
                                                            pretrain= pretrain)
 
        self.decoder = PSPHEAD(in_channels=1024, out_channels=512, pool_sizes = [1, 2, 3, 6], num_classes=self.num_classes)
        self.cls_seg = nn.Sequential(
            nn.Conv2d(512, self.num_classes, kernel_size=3, padding=1),
        )
        
    def forward(self, x):
        _, _, h, w = x.size()
        c1,c2,c3,c4,c5 = self.backbone(x) 
        #c1 64/1
        #c2 128/2
        #c3 256/4
        #c4 512/8
        #c5 2048/16
        # print(c5.shape)
        x = self.decoder(c5)
        x = nn.functional.interpolate(x, size=(h, w),mode='bilinear', align_corners=True)
        out_cls = self.cls_seg(x)
        return {'cls':out_cls, 'side':[]}
 
 
if __name__ == "__main__":
    model = Pspnet(num_classes=33)
    model = model.cuda()
    a = torch.ones([2, 3, 224, 224])
    a = a.cuda()
    print(model(a).shape)