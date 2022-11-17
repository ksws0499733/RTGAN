import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import numpy as np
# import LovaszLoss as lloss
import torch.nn.functional as F


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def create_mask(width, height, mask_width, mask_height, x=None, y=None):
    mask = np.zeros((height, width))
    mask_x = x if x is not None else random.randint(0, width - mask_width)
    mask_y = y if y is not None else random.randint(0, height - mask_height)
    mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
    return mask

def stitch_images(inputs, *outputs, img_per_row=2):
    gap = 5
    columns = len(outputs) + 1

    width, height = inputs[0][:, :, 0].shape
    img = Image.new('RGB', (width * img_per_row * columns + gap * (img_per_row - 1), height * int(len(inputs) / img_per_row)))
    images = [inputs, *outputs]

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img

def imshow(img, title=''):
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    plt.axis('off')
    plt.imshow(img, interpolation='none')
    plt.show()

def imsave(img, path):
    im = Image.fromarray(img.cpu().numpy().astype(np.uint8).squeeze())
    im.save(path)

class Progbar(object):
    """Displays a progress bar.

    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=25, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules or
                                 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.

        Arguments:
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)

class Gradient(object):
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
        self.expanded_padding = (gauss_kernel_size)//2

    def gauss_grad(self, target):
        # target = target.unsqueeze(1) # B,1,H,W
        target_g = F.conv2d(target,                                                     
                            self.weight_g, 
                            padding=self.expanded_padding)

        target_direct, target_gradient = self._get_gradient(target_g)

        return target_direct, target_gradient

    def nogauss_grad(self, target):
        # target = target.unsqueeze(1) # B,1,H,W
        target_g = target
        target_direct, target_gradient = self._get_gradient(target_g)

        return target_direct, target_gradient

    def _get_gradient(self,x):
        grad_x = F.conv2d(F.pad(x,(1,1,1,1)), 
                        self.weight_x)
        grad_y = F.conv2d(F.pad(x,(1,1,1,1)), 
                        self.weight_y)

        direct = torch.atan2(grad_x, grad_y)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)

        return direct, gradient
    def gauss_gradxy(self, target):
        # target = target.unsqueeze(1) # B,1,H,W
        target_g = F.conv2d(target,                                                     
                            self.weight_g, 
                            padding=self.expanded_padding)
                            
        grad_x, grad_y = self._get_gradientxy(target_g)

        return grad_x, grad_y

    def nogauss_gradxy(self, target):
        # target = target.unsqueeze(1) # B,1,H,W
        target_g = target
        grad_x, grad_y = self._get_gradientxy(target_g)

        return grad_x, grad_y

    def _get_gradientxy(self,x):
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
