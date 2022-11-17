import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class conv_block(nn.Module):
    def __init__(self,in_c,out_c):
        super(conv_block,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_c,out_c,kernel_size=(3,3),stride=1,padding=1,padding_mode='reflect'),
            nn.BatchNorm2d(out_c),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=(3, 3), stride=1, padding=1, padding_mode='reflect',bias = False),
            nn.BatchNorm2d(out_c),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class Downsample(nn.Module):
    def __init__(self,channel):
        super(Downsample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=2, padding=1,  bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

    def forward(self,x):
        return self.layer(x)


class Upsample(nn.Module):
    def __init__(self,channel):
        super(Upsample, self).__init__()
        self.conv1 = nn.Conv2d(channel,channel//2,kernel_size=(1,1),stride=1)

    def forward(self,x,featuremap):
        x = F.interpolate(x,scale_factor=2,mode='nearest')
        x = self.conv1(x)
        x = torch.cat((x,featuremap),dim=1)
        return x

class UNET(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(UNET, self).__init__()
        self.layer1 = conv_block(in_channel,out_channel)
        self.layer2 = Downsample(out_channel)
        self.layer3 = conv_block(out_channel,out_channel*2)
        self.layer4 = Downsample(out_channel*2)
        self.layer5 = conv_block(out_channel*2,out_channel*4)
        self.layer6 = Downsample(out_channel*4)
        self.layer7 = conv_block(out_channel*4,out_channel*8)
        self.layer8 = Downsample(out_channel*8)
        self.layer9 = conv_block(out_channel*8,out_channel*16)
        self.layer10 = Upsample(out_channel*16)
        self.layer11 = conv_block(out_channel*16,out_channel*8)
        self.layer12 = Upsample(out_channel*8)
        self.layer13 = conv_block(out_channel*8,out_channel*4)
        self.layer14 = Upsample(out_channel*4)
        self.layer15 = conv_block(out_channel*4,out_channel*2)
        self.layer16 = Upsample(out_channel*2)
        self.layer17 = conv_block(out_channel*2,out_channel)
        self.layer18 = nn.Conv2d(out_channel,out_channel,kernel_size=(1,1),stride=1)
        self.act = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        x = self.layer3(x)
        f2 = x
        x = self.layer4(x)
        x = self.layer5(x)
        f3 = x
        x = self.layer6(x)
        x = self.layer7(x)
        f4 = x
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x,f4)
        x = self.layer11(x)
        x = self.layer12(x,f3)
        x = self.layer13(x)
        x = self.layer14(x,f2)
        x = self.layer15(x)
        x = self.layer16(x,f1)
        x = self.layer17(x)
        x = self.layer18(x)
        cls_out =  self.act(x)

        # print(cls_out.shape)
        return {'cls':cls_out, 
                'side':[]}


if __name__ == '__main__':
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(10,3,256,256)
    model = UNET(3,64)
    #if hasattr(torch.cuda, 'empty_cache'):
        #torch.cuda.empty_cache()

    x = model(x)
    print(x.size())

    wiriter = SummaryWriter('log1')
    wiriter.add_graph(model,x)
