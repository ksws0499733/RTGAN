# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import os
import numpy as np
from dataloaders import custom_transforms as tr
from torch.utils import data
from mypath import Path
import cv2.cv2 as cv2

class iRailway(data.Dataset):
    def __init__(self, img_folder, ann_folder=None, 
                        label_folder=None, 
                        transforms=None, 
                        return_masks=True, 
                        split='train',
                        repeat = 10):

        if not isinstance(img_folder, list):
            img_folder = [img_folder,]

        self.split = split

        self.img_list = []
        self.ann_list = []
        self.cls_list = []
        self.ins_list = []
        self.mask_list = []
        print('    ',split)
        for img_fd in img_folder:
            self.load_from_subdir(img_fd, split, 
                                            repeat,
                                            ann_folder,
                                            label_folder)

            print('\t',img_fd,' : ', len(self.img_list))

        self.transforms = transforms
        self.return_masks = return_masks

    def load_from_subdir(self, img_folder,
                                split,
                                repeat,
                                ann_folder=None,
                                label_folder=None):
        img_list = os.listdir(img_folder)
        if split == 'test':
            for imgfile in img_list:
                if imgfile.endswith('.jpg'):
                    clsfile = imgfile.replace('.jpg','_cls.png')
                    maskfile = imgfile.replace('.jpg','_mask.png')
                    insfile = imgfile.replace('.jpg','_ins.png')

                    self.img_list.append(os.path.join(img_folder,imgfile))
                    self.ins_list.append(os.path.join(img_folder,insfile))
                    self.cls_list.append(os.path.join(img_folder,clsfile))
                    self.mask_list.append(os.path.join(img_folder,maskfile))
        elif split in ['train','val']:

            if ann_folder is None:
                ann_folder = os.path.join(img_folder,"Info")

            if os.path.isdir(ann_folder): 
                if label_folder is None:
                    label_folder = os.path.join(ann_folder,"class_mask_png")        
                ann_list = os.listdir(ann_folder)
                label_list = os.listdir(label_folder)

                for annfile in ann_list:
                    if annfile.endswith('.json'):
                        # name,ext = os.path.splitext(annfile)
                        imgfile = annfile.replace('.json', '.jpg')
                        clsfile = annfile.replace('.json', '_cls.png')
                        insfile = annfile.replace('.json', '_ins.png')
                        if imgfile in img_list and clsfile in label_list and insfile in label_list:
                            for i in range(repeat):                    
                                self.img_list.append(os.path.join(img_folder,imgfile))
                                self.cls_list.append(os.path.join(label_folder,clsfile))
                                self.ins_list.append(os.path.join(label_folder,insfile))

                                with open(os.path.join(ann_folder,annfile), 'r') as f:
                                    ann_info = json.load(f)
                                    self.ann_list.append(ann_info)
            else:
                label_folder = os.path.join(img_folder,split)
                if os.path.isdir(label_folder):
                    label_list = os.listdir(label_folder)
                    for file in label_list:
                        if file.endswith('.jpg'):
                            imgfile = file
                            clsfile = file.replace('.jpg', '_cls.png')
                            insfile = file.replace('.jpg', '_ins.png') 
                            mskfile = file.replace('.jpg', '_mask.png')       
                            self.img_list.append(os.path.join(label_folder,imgfile))
                            self.cls_list.append(os.path.join(label_folder,clsfile))
                            self.ins_list.append(os.path.join(label_folder,insfile))
                            self.mask_list.append(os.path.join(label_folder,mskfile))

    def __getitem__(self, idx):

        img_path = self.img_list[idx]        
        ins_path = self.ins_list[idx]
        cls_path = self.cls_list[idx]
        msk_path = self.mask_list[idx]



        _img = cv2.imread(img_path)
        if _img is None:
            return self.image_error(idx)


        if(os.path.isfile(cls_path)):
            _cls = cv2.imread(cls_path) 
        else:
            _cls = np.zeros_like(_img)
        _cls = _cls > 2
        _cls = _cls.astype(np.float32)

        if(os.path.isfile(msk_path)):
            _msk = cv2.imread(msk_path) 
        else:
            _msk = np.zeros_like(_img)

        ids = np.array([0])

        if(os.path.isfile(ins_path)):
            _ins = cv2.imread(ins_path) 
        else:
            _ins = np.zeros_like(_img)
        _ins = rgb2id(_ins)    
        _ins = _ins == ids[:, None, None]
        _ins = _ins.astype(np.uint8)

        img = _img
        clss = _cls[:,:,0]
        inss = _ins
        mask = _msk[:,:,0]
        sample = {'image': img, 'label': clss,'instence': inss,'mask':mask}
        return self.transforms(sample)


    def image_error(self, idx = 0):

        print("Image idx:{} is error image; path:{}".format(idx, self.img_list[idx]))
        sample = {'image': None, 'label': None,'instence': None,'mask':None}
        if self.transforms is None:
            return self.transform_val(sample)
        else:
            return self.transforms(sample)

    def transform_val(self, sample):
        composed_transforms = tr.Compose([
            tr.crop_auto(0,0,512,512),
            tr.ToTensor()])
        return composed_transforms(sample)

    def __len__(self):
        return len(self.img_list)

    def get_height_and_width(self, idx):
        ann_info = self.ann_list[idx]
        height = ann_info['imageHeight']
        width = ann_info['imageWidth']
        return height, width

def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 2] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 0]
    return int(color[1] + 256 * color[1] + 256 * 256 * color[0])

def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color

import torch.utils.data as tdata
import json
def buildA(args,image_set='iRailway',split='train'):

    root,cfg = Path.db_root_dir(image_set)


    with open(cfg,'r') as f:
        dataset_root_list = f.readlines()
        
    print(dataset_root_list)
    dataset_root = []
    for dataset_r in dataset_root_list:
        pth = os.path.join(root,dataset_r[:-1])# remove '\n' in the end
        if os.path.isdir(pth):
            dataset_root.append(pth)  
    PATHS = {
        "train": dataset_root,
        "val": dataset_root,
        "test": args.testPath
    }

    transform_dict = {
        "affine": tr.RandomAffine(),
        "GaussBlur": tr.RandomGaussianBlur(),
        "crop": tr.crop_auto(0,0,512,512),
        "dig": tr.RandomDig(),
        "line": tr.RandomDigLine(),
        "shadow": tr.RandomShadow(),
        "shadowLine": tr.RandomShadowLine()

    }


    tr_list = []
    if args.dataAug is not 'no':
        print(args.dataAug)
        dataAug_list = args.dataAug.split(',')
        for aug in dataAug_list:
            if aug in transform_dict.keys():
                tr_list.append(transform_dict[aug])
    

    tr_list.append(tr.ToTensor())

    transforms=tr.Compose(tr_list)


    img_folder = PATHS[split]
    dataset = iRailway(img_folder,
                        split=split,
                        transforms=transforms,
                        repeat= args.dataRepeat
                        )

    return dataset


if __name__ == '__main__':
    image_set = "train"
    img_folder = r"E:\dataSet\all_dataset\nanjing01"
    dataset = iRailway(img_folder,
                        transforms =  tr.Compose([
                         tr.RandomShadowLine(),
                         tr.RandomShadow(),
                         tr.RandomDig(),
                         tr.RandomAffine(),
                         tr.ToTensor()
                        ])
                    )

    from torch.utils.data import DataLoader
    from dataloaders.utils import collate_fn
    # from models.my_marcher_globelpanic import targer_cat, mask_iou_cost_dice
    data_loader_train = DataLoader(dataset, batch_size=3,collate_fn=collate_fn)
    for sample in data_loader_train:
        # print(sample)
        img = sample['image']    # H*W*3
        mask = sample['label']   # H*W
        inss = sample['instence']  # K*H*W
        print("img",img.shape)
        print("mask",mask.shape)
        print("inss",inss.shape)
        img_show = img.numpy()
        img_show = img_show[0]
        img_show = img_show.transpose((1, 2, 0))

        cls_show = mask.numpy()[0]
        
        cv2.imshow("img_show1",img_show/255)
        cv2.imshow("img_show2",cls_show)
        cv2.waitKey(0)

    
