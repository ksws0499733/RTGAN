# from matplotlib import transforms
from ast import Raise
import torch
import numpy.random as random
import numpy as np
import cv2.cv2 as cv2
import math

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        inss = sample['instence']
        selec = sample['mask']
        img = img.astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        sample.update({'image': img,
                'label': mask,
                'instence': inss,
                'mask':selec})

        return sample

# sample = {'image': img, 'label': clss,'instence': inss}
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        inss = sample['instence']
        selec = sample['mask']

        if img is not None:
            img = img.astype(np.float32).transpose((2, 0, 1)) # C,H,W
            mask = mask.astype(np.float32)
            inss = inss.astype(np.float32)#.transpose((2, 0, 1))# K,H,W

            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).float()
            inss = torch.from_numpy(inss).float()
            selec = torch.from_numpy(selec).float()
            sample.update({'image': img,
                    'label': mask,
                    'instence': inss,
                    'mask':selec})

        return sample

class RandomHorizontalFlip(object):
    def __init__(self, level=1.0):
        self.level = level

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        inss = sample['instence']
        selec = sample['mask']
        if img is not None:
            if random.random() < 0.5:
                img = np.flip(img, 0)
                mask = np.flip(mask, 0)
                inss = np.flip(inss, 0)

            sample.update({'image': img,
                    'label': mask,
                    'instence': inss,
                    'mask':selec})

        return sample

class RandomAffine(object):

    def __init__(self, degrees=10, translate=.1, scale=.1, shear=10, border=(0, 0)):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.border = border


    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        inss = sample['instence']
        selec = sample['mask']
        if img is not None:
            height = img.shape[0] + self.border[0] * 2  # shape(h,w,c)
            width = img.shape[1] + self.border[1] * 2
            # print(height,width)

            # Rotation and Scale
            R = np.eye(3)
            a = random.uniform(-self.degrees, self.degrees)
            a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
            s = random.uniform(1 - self.scale, 1 + self.scale)
            # s = 2 ** random.uniform(-scale, scale)
            R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

            # Translation
            T = np.eye(3)
            T[0, 2] = random.uniform(-self.translate, self.translate) * img.shape[1] + self.border[1]  # x translation (pixels)
            T[1, 2] = random.uniform(-self.translate, self.translate) * img.shape[0] + self.border[0]  # y translation (pixels)

            # Shear
            S = np.eye(3)
            S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
            S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

            # Combined rotation matrix
            M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
            if (self.border[0] != 0) or (self.border[1] != 0) or (M != np.eye(3)).any():  # image changed
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
                mask = cv2.warpAffine(mask, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR,borderValue= 0)
                inss_dct = np.zeros_like(inss)
                for i in range(inss.shape[0]):
                    inss_dct[i] = cv2.warpAffine(inss[i], M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR,borderValue= 0)
                inss = inss_dct
                if selec is not None:
                    selec = cv2.warpAffine(selec, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR,borderValue= 0)
                

            sample.update({'image': img,
                    'label': mask,
                    'instence': inss,
                    'mask':selec})

        return sample

class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        inss = sample['instence']
        if img is None:
            return sample
        
        if random.random() < 0.5:
            k = random.randint(0,17)*2 + 1
            s = random.randint(0,k)
            img = cv2.GussianBlur(img, 
                                ksize=(k, k), 
                                sigmaX=s)
        sample.update({'image': img,
                'label': mask,
                'instence': inss})

        return sample

class RandomShadowLine(object):
    def __init__(self, lineId = 3,level=1.0):
        self.lineID = lineId
        self.level = level

    def __call__(self, sample):
        img = sample['image']
        label = sample['label']

        if img is None:
            return sample       

        img = img.astype(np.float32)

        h,w,c = img.shape

        if random.random() < 0.3:
            
            img_mask0 = label > (self.lineID-1)
            img_mask0 = img_mask0.astype(np.float32)

            bak = random.rand(h,w,c)*255
            mask = img_mask0 + mask
            img = img* (1-img_mask0)[:,:,None] + bak*img_mask0[:,:,None]
        
        img = img.astype(np.uint8)
        mask = (mask>0).astype(np.float32)

        sample.update({'image': img})

        return sample

class RandomShadow(object):
    def __init__(self,prob=1.0, shadow_ratio=0.5):
        self.prob = prob
        self.shadow_ratio = shadow_ratio

    def __call__(self, sample):
        img = sample['image']
       
        if img is None:
            return sample
        img = img.astype(np.float32)
        
        if random.random() < self.prob:
            h = img.shape[0]
            w = img.shape[1]            

            xc = int(random.random()*w)
            yc = int(random.random()*h)
            ag = random.random()*math.pi*2
            nx, ny = np.cos(ag), np.sin(ag)

            k = list(np.array([-xc, w-xc, -yc, h-yc])/ np.array([nx, nx, ny, ny]))
            k.sort(key=self.abs_sort)
            keyPoints = [(xc+k[0]*nx,yc + k[0]*ny),
                         (xc+k[1]*nx,yc + k[1]*ny),
                        ]
            cPoints = [(0,0),(w,0),(w,h),(0,h)]
            
            for pt in cPoints:
                if ny*(pt[0]-xc)-nx*(pt[1]-yc) >0:
                    keyPoints.append(np.array(pt))
            
            keyPoints = np.array(keyPoints, dtype=np.int32)
            keyPoints = cv2.convexHull(keyPoints)

            img_mask = np.zeros((h,w), dtype=np.uint8)+255
            
            
            img_mask = cv2.fillConvexPoly(img_mask,keyPoints, self.shadow_ratio*255)
            img_mask = img_mask.astype(np.float32)/255
            img = img*img_mask[:,:,None]
        
        img = img.astype(np.uint8)

        sample.update({'image': img})

        return sample

    def abs_sort(self,x):
        return math.fabs(x)

class RandomDigLine(object):
    def __init__(self, lineId = 3):
        self.lineID = lineId

    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        inss = sample['instence']
        mask = sample['mask'].astype(np.float64)
        if img is None:
            return sample       

        img = img.astype(np.float32)
        
        img, mask = self._digLine(img, mask, repeat= random.randint(3))
        img, mask = self._digCircle(img, mask, repeat= random.randint(3))
        
        img = img.astype(np.uint8)
        # img = img_mask

        sample.update({'image': img,
                'label': label,
                'instence': inss,
                'mask':mask})

        return sample

    def _digLine(self,img,mask,repeat=1):
        h,w,c = img.shape
        xc = int(random.random()*w)
        yc = int(random.random()*h)

        Xc = random.randint(0,w,size=(repeat,))
        Yc = random.randint(0,h,size=(repeat,))
        Ag = random.random(size=(repeat,))*math.pi
        Len0 = random.randint(0,h,size=(repeat,))

        for xc,yc,ag,len0 in zip(Xc,Yc,Ag,Len0):

            nx = int(np.cos(ag)*len0)
            ny = int(np.sin(ag)*len0)
            s = (xc + nx,yc + ny)
            e = (xc - nx,yc - ny)

            color = (0,0,0) if random.random()>0.5 else (255,255,255)
            thickness = int(random.random()*6)+3

            cv2.line(img,s,e,color,thickness)
            cv2.line(mask,s,e,1,thickness)

        return img, mask

    def _digCircle(self,img, mask, repeat=1):
        h,w,c = img.shape
        xc = int(random.random()*w)
        yc = int(random.random()*h)

        Xc = random.randint(0,w,size=(repeat,))
        Yc = random.randint(0,h,size=(repeat,))
        Ag = random.random(size=(repeat,))*math.pi*2
        Len0 = random.randint(h,h*3,size=(repeat,))

        for xc,yc,ag,len0 in zip(Xc,Yc,Ag,Len0):

            nx = int(np.cos(ag)*len0)
            ny = int(np.sin(ag)*len0)
            s = (xc+nx,yc+ny)

            color = (0,0,0) if random.random()>0.5 else (255,255,255)
            thickness = int(random.random()*6)+3

            cv2.circle(img,s,len0,color,thickness)
            cv2.circle(mask,s,len0,1,thickness)

        return img, mask

class RandomDig(object):
    def __init__(self, sque_ratio = 0.5, hline_ratio = 0.25, hole_ratio = 0.12):
        self.sque_ratio = sque_ratio
        self.hline_ratio = hline_ratio
        self.hole_ratio = hole_ratio

    def __call__(self, sample):
        img = sample['image']
        mask = sample['mask']
        if img is None:
            return sample
        img = img.astype(np.float32)
        mask0 = np.zeros_like(img)[:,:,0]
        selct = random.random()
        if random.random() < 0.33:
            img,mask1 = self.dig_hole(img)
            mask0 += mask1

        elif selct <= 0.66:
            img,mask2 = self.dig_hline(img)
            mask0 += mask2
        else:
            img,mask3 = self.dig_sque(img)
            mask0 += mask3

        img = img.astype(np.uint8)
        mask[mask0>0] = 255
        sample.update({'image': img,
                        'mask': mask
                        })

        return sample

    def dig_hole(self, img):
        h,w,c = img.shape
        rr =  random.randint(w*self.hole_ratio*0.9,w*self.hole_ratio*1.1)
        xc = random.randint(rr/2,w-rr/2)
        yc = random.randint(rr/2,h-rr/2)

        mask = np.zeros((h,w))
        cv2.circle(mask, (xc,yc), rr, 255, -1)
        img[mask > 0,:] = 0

        return img,mask

    def dig_hline(self, img):
        h,w,c = img.shape
        ht = int(h*self.hline_ratio)
        yc = random.randint(0,h-ht)
        yd = min(yc+ht,h)
        mask = np.zeros((h,w))

        img[yc:yd] = 0  
        mask[yc:yd] = 255
        return img, mask

    def dig_sque(self, img, mH = 20, mW = 20):

        sx,sy = random.randint(5,mW),random.randint(5,mH)

        h, w, c = img.shape
        mask = np.zeros((h,w))
        for ii in range(w//sx):
            for jj in range(h//sx):

                if random.random() < self.sque_ratio:
                    x1a = min(ii*sx, w)
                    y1a = min(jj*sx, h)
                    x2a = min(ii*sx + sx , w)
                    y2a = min(jj*sx + sx , h)

                    img[y1a:y2a,x1a:x2a] = 0
                    mask[y1a:y2a,x1a:x2a] = 255
        return img,mask

class resize(object):
    def __init__(self, w = 480, h=None):
        self.w = w
        self.h = h or w
        self.shape = (int(self.w),int(self.h))

    def __call__(self, sample):
        img = sample['image']    # H*W*3
        mask = sample['label']   # H*W
        inss = sample['instence']  # K*H*W
        selec = sample['mask']
        if img is None:
            return sample
        img = cv2.resize(img, self.shape)
        mask = cv2.resize(mask, self.shape)

        _inss = []
        for ins in inss:
            _ins = cv2.resize(ins, self.shape)
            _inss.append(_ins)
        inss = np.stack(_inss,axis=0)

        selec = cv2.resize(selec, self.shape)


        sample.update({'image': img,
        'label': mask,
        'instence': inss,
        'mask':selec})
        return sample

class crop(object):
    def __init__(self,x=0, y=0, w = 480, h=None):
        self.x = x
        self.y = y
        self.w = w
        self.h = h or w
        self.shape = (int(self.h),int(self.w))

    def __call__(self, sample):
        img = sample['image']    # H*W*3
        mask = sample['label']   # H*W
        inss = sample['instence']  # K*H*W
        selec = sample['mask']
        if img is None:
            return sample
        h,w,c = img.shape
        y1 = self.y
        y2 = min(self.y+self.h, h)
        x1 = self.x
        x2 = min(self.x+self.w, w)
        img = img[y1:y2,x1:x2]
        mask = mask[y1:y2,x1:x2]
        inss = inss[y1:y2,x1:x2]
        selec = selec[y1:y2,x1:x2]

        sample.update({'image': img,
        'label': mask,
        'instence': inss,
        'mask':selec})
        return sample

class crop_auto(object):
    def __init__(self,x=0, y=0, w = 512, h=None):
        self.x = x
        self.y = y
        self.w = w
        self.h = h or w
        self.shape = (int(self.h),int(self.w))

    def __call__(self, sample):
        if sample['image'] is None:
            return sample
        img = self.extend(sample['image'],type='hwc')    # H*W*3
        mask = self.extend(sample['label'],type='hw')   # H*W
        inss = self.extend(sample['instence'],type='chw')  # K*H*W
        selec = self.extend(sample['mask'],type='hw',bg=1)
        # print('1',img.shape)
        h,w,c = img.shape
        y1 = np.random.randint(h - self.h)
        y2 = y1+self.h
        x1 = np.random.randint(w - self.w)
        x2 = x1+self.w
        img = img[y1:y2,x1:x2]
        mask = mask[y1:y2,x1:x2]
        inss = inss[:,y1:y2,x1:x2]
        selec = selec[y1:y2,x1:x2]
        # print('y1',y1,y2,y2-y1)
        # print('x1',x1,x2,x2-x1)
        # print('2',img.shape)


        sample.update({'image': img,
        'label': mask,
        'instence': inss,
        'mask':selec})
        return sample

    def extend(self, img, type='hwc', bg=0, ext_scale = 0.25):
        dw = self.w
        dh = self.h

        new_w = int(dw*(ext_scale)) 
        new_h = int(dh*(ext_scale)) 
        s_x = int(dw*ext_scale/2)
        s_y = int(dh*ext_scale/2)

        if type == 'hwc':
            h,w,c = img.shape
            eimg = np.zeros((new_h+h,new_w+w,c))+bg
            eimg[s_y:s_y+h, s_x:s_x+w] = img

        elif type == 'chw':
            c,h,w = img.shape
            eimg = np.zeros((c,new_h+h,new_w+w))+bg
            eimg[:,s_y:s_y+h, s_x:s_x+w] = img 
        elif type == 'hw':
            h,w = img.shape
            eimg = np.zeros((new_h+h,new_w+w))+bg
            eimg[s_y:s_y+h, s_x:s_x+w] = img 
        else:
            raise RuntimeError()

        return eimg


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


if __name__ == '__main__':

    img = cv2.imread(r'doc/dji000045.jpg')  
    mask = cv2.imread(r'doc/dji000045_cls.png')[:,:,0]
    inss = cv2.imread(r'doc/dji000045_ins.png')  


    sample = {'image': img, 'label': mask,'instence': inss}

    transforms = Compose([
        # RandomShadowLine(),
        # RandomShadow(),
        RandomDig(),
        # RandomAffine(),        
        ])


    sample = transforms(sample)
    img = sample['image']
    mask = sample['label']
    inss = sample['instence']
    cv2.imshow("out",img)

    mask = mask > 2
    mask = mask.astype(np.float32)
    cv2.imshow("mask",mask)
    cv2.waitKey(0)


