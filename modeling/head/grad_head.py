import torch
import torch.nn as nn
import torch.nn.functional as F

class gradNet(nn.Module):
    def __init__(self):
        super(gradNet, self).__init__() 


        device = torch.device('cuda')
        kernel_x = [[-1., 0., 1.],
                    [-2., 0., 2.],
                    [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device=device)

        kernel_y = [[-1., -2., -1.],
                    [ 0.,  0.,  0.],
                    [ 1.,  2.,  1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device=device)

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    
    def forward(self, x):
        grad_x = F.conv2d(x,self.weight_x, padding=1)
        grad_y = F.conv2d(x,self.weight_y, padding=1)


        direct = torch.atan2(grad_x, grad_y)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)

        return direct, gradient

    