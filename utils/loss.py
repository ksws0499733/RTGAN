import torch
import torch.nn as nn
import numpy as np
# import LovaszLoss as lloss
import torch.nn.functional as F


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
 

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        print("build_loss:",mode)
        
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'Lova':
            return self.LovaLoss
        elif mode == 'bce':
            # return nn.BCEWithLogitsLoss(reduction='mean')
            return binary_cross_entropy.apply
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        # logit = F.softmax(logit,dim=1)
        # print(target.shape)
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def LovaLoss(self, logit, target):
        return lovasz_softmax(logit, target, classes='present', per_image=False, ignore=None)

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        logit = F.softmax(logit,dim=1)
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

class binary_cross_entropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target):
        

        ignoreIdx = 255
        mask = (target == ignoreIdx).float()
        target = target*(1-mask)
        input = torch.sigmoid(input)*(1-mask)

        beta = 1 - torch.mean(target)
        weights = 1 - beta + (2*beta -1)*target
        pos = (input >=0.).float()
        bce_loss = torch.log(1 + (input -2*input*pos).exp()) - input*(target - pos)
        loss = torch.mean((bce_loss*weights).view(-1), dim=0, keepdim=True)
        ctx.save_for_backward(input, target)
        return loss
    @staticmethod
    def backward(ctx, grad_output):
        input, target, = ctx.saved_variables
        beta = 1 - torch.mean(target)
        weights = 1 - beta + (2*beta -1)*target
        grad = (input - target)*weights
        return grad*grad_output, None



class InstanceLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False, sigma = 2.0, pPara = 1.0):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
        self.sigma = sigma
        self.pushPara = pPara
 

    def build_loss(self, mode='ins'):
        if mode == 'ins':
            return self.InsLoss
        else:
            raise NotImplementedError

    def pull(self, pred, target):
        #pred --- n,c,h,w -->   n,   c,1[k],hw
        #target --- n,k,h,w --> n,1[c],   k,hw       
        pred = torch.unsqueeze(pred, dim=2).flatten(start_dim=3)
        target = torch.unsqueeze(target, dim=1).flatten(start_dim=3)

        n1, c, valid , hw1 = pred.size()
        n2, valid, k , hw2 = pred.size()

        assert n1 == n2 and hw1 == hw2
        n = n1
        # hw = hw1

        A = pred*target
        A_bar = torch.sum(A, dim=3, keepdim=True) / torch.sum(target, dim=3, keepdim=True)

        A2 = (A-A_bar)**2 * target
        A2_mse = torch.sum(A2, dim=3, keepdim=True) / torch.sum(target**2, dim=3, keepdim=True)
        loss = torch.sum(A2_mse) / c / k 

        # criterion = nn.MSELoss(reduce=True, size_average=True)
        # criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
        #                                 size_average=self.size_average)
        # if self.cuda:
        #     criterion = criterion.cuda()
        # loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n
        return loss

    def push(self, pred, target):
        #pred --- n,c,h,w -->   n,   c,1[k],hw
        #target --- n,k,h,w --> n,1[c],   k,hw       
        pred = torch.unsqueeze(pred, dim=2).flatten(start_dim=3)
        target = torch.unsqueeze(target, dim=1).flatten(start_dim=3)

        n1, c, valid , hw1 = pred.size()
        n2, valid, k , hw2 = pred.size()
        assert n1 == n2 and hw1 == hw2
        n = n1
        A = pred*target
        A_bar = torch.sum(A, dim=3, keepdim=True) / torch.sum(target, dim=3, keepdim=True)

        A3 =  (A_bar - A_bar.transpose(2, 3))**2
        A3_exp = torch.exp( - torch.sum(A3, dim=1, keepdim=True) / 2/self.sigma**2 )

        loss = torch.sum(A3_exp) / k /k

        if self.batch_average:
            loss /= n
        return loss

    def InsLoss(self, pred, target):
        # loss_pull = self.pull(pred, target)
        # loss_push = self.push(pred, target)

        #pred --- n,c,h,w -->   n,   c,1[k],hw
        #target --- n,k,h,w --> n,1[c],   k,hw       
        pred = torch.unsqueeze(pred, dim=2).flatten(start_dim=3)
        target = torch.unsqueeze(target, dim=1).flatten(start_dim=3)

        n1, c, valid , hw1 = pred.size()
        n2, valid, k , hw2 = pred.size()
        assert n1 == n2 and hw1 == hw2
        n = n1
        A = pred*target
        A_bar = torch.sum(A, dim=3, keepdim=True) / torch.sum(target, dim=3, keepdim=True)

        A2 = (A-A_bar)**2 * target
        A2_mse = torch.sum(A2, dim=3, keepdim=True) / (torch.sum(target, dim=3, keepdim=True) + 1e-6)**2     
        loss_pull = torch.sum(A2_mse) / c / k    

        A3 =  (A_bar - A_bar.transpose(2, 3))**2
        A3_exp = torch.exp( - torch.sum(A3, dim=1, keepdim=True) / 2/self.sigma**2 )

        loss_push = torch.sum(A3_exp) / k /k
        loss = loss_pull + loss_push*self.pushPara

        if self.batch_average:
            loss /= n
        return loss

class GradientLosses(object):
    def __init__(self, weight=None, 
                        size_average=True, 
                        batch_average=True, 
                        ignore_index=255, 
                        cuda=False,
                        gauss_kernel_size = 17,
                        gauss_kernel_sigma = 2.0, 
                        pPara = 1.0):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
        device = torch.device('cuda')
        kernel_x = [[-1., 0., 1.],
                    [-2., 0., 2.],
                    [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device=device)

        kernel_y = [[-1., -2., -1.],
                    [ 0.,  0.,  0.],
                    [ 1.,  2.,  1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device=device)


        guass_kernel = self._create_gauss_kernel(gauss_kernel_size,gauss_kernel_sigma)
        guass_kernel = torch.FloatTensor(guass_kernel).unsqueeze(0).unsqueeze(0).to(device=device)

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)
        self.weight_g = nn.Parameter(data=guass_kernel, requires_grad=False)
        self.expanded_padding = ((gauss_kernel_size)//2,
                                (gauss_kernel_size)//2,
                                (gauss_kernel_size)//2,
                                (gauss_kernel_size)//2)

    def build_loss(self, mode='rawInput'):
        if mode == 'rawInput':
            self.isRawInput = True
            return self.GradLoss
        if mode == 'gradInput':
            self.isRawInput = False
            return self.GradLoss
        else:
            raise NotImplementedError

    def GradLoss(self, pred, target):
        #pred --- B,C,H,W
        #traget -- B,H,W
        B,C,H,W = pred.shape
        target = target.unsqueeze(1) # B,1,H,W
        target_g = F.conv2d(F.pad(target, 
                                self.expanded_padding,
                                mode='constant',
                                value=0
                                ),                        
                        self.weight_g)

        # print(target.shape,target_g.shape)

        assert target.shape == target_g.shape

        mask = torch.zeros_like(target_g)
        # mask[target<1] = 1   # ignore track region inner
        mask[target_g>0.01] = 1 

        # if self.isRawInput:
        #     pred_direct, _ = self._get_gradient(pred[:,1:])
        # else:
        #     pred_direct = pred
        # target_direct, _ = self._get_gradient(target_g)
        # loss = ((1-torch.cos(target_direct - pred_direct))*mask).sum()


        pred_gx, pred_gy = self._get_gradientXY(pred[:,1:])
        target_gx, target_gy = self._get_gradientXY(target_g)

        # loss =torch.pow((torch.atan2(target_gx,target_gy) - torch.atan2(pred_gx,pred_gy)),2).sum()

        loss = ((torch.pow(target_gx-pred_gx, 2) + torch.pow(target_gy-pred_gy, 2))*mask ).sum()/(1+mask.sum())

        

        # if self.size_average:
        #     loss /= (H*W)

        if self.batch_average:
            loss /= B

        return loss


    def _get_gradient(self,x):
        grad_x = F.conv2d(F.pad(x,(1,1,1,1)), 
                        self.weight_x)
        grad_y = F.conv2d(F.pad(x,(1,1,1,1)), 
                        self.weight_y)

        direct = torch.atan2(grad_x, grad_y)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)

        return direct, gradient

    def _get_gradientXY(self,x):
        grad_x = F.conv2d(F.pad(x,(1,1,1,1)), 
                        self.weight_x)
        grad_y = F.conv2d(F.pad(x,(1,1,1,1)), 
                        self.weight_y)

        return grad_x, grad_y

        

    def _create_gauss_kernel(self, size, sigma=1.0):
        if sigma == 0:
            return np.ones(size)
        else:
            sigma3 = sigma*3
            if isinstance(size, list):
                h,w = size
            else:
                h = w =size

            X = np.linspace(-sigma3, sigma3, w)
            Y = np.linspace(-sigma3, sigma3, h)
            y,x = np.meshgrid(Y,X)
            gauss =  np.exp(-(x**2 + y**2)/ (2*sigma**2)) / ( 2*np.pi * sigma**2)
            gauss = gauss/gauss.sum()
            return gauss

# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


def build_loss():
    pass


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(2, 3, 5, 7).cuda()
    b = torch.floor(torch.rand(2, 5, 7)*3).cuda()

    print(loss.my_index_IOU(a,b).cpu().numpy())
    # print(loss.CrossEntropyLoss(a, b).item())
    # print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    # print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




