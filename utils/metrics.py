import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0

        self.precision=[]
        self.recall = []

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return Acc
    
    def Pixel_Recall_class(self):
        Rec = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        return Rec

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + 
                    np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        # MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + 
                    np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))


        FWIoU = (freq * iu)
        return FWIoU

    def FScore(self):
        Fscore = np.diag(self.confusion_matrix)*2 / (
                    np.sum(self.confusion_matrix, axis=1) + 
                    np.sum(self.confusion_matrix, axis=0))

        return Fscore        

    def _generate_matrix(self, gt_image, pre_image, mask):
        # mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]#真值*类别数 + 预测值
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image, mask = None):
        assert gt_image.shape == pre_image.shape
        _mask = mask or ((gt_image >= 0) & (gt_image < self.num_class))
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image, _mask)
        # self._add_batch( gt_image, pre_image)
        # self._add_batch2( gt_image, pre_image)
    
    
    def _add_batch2(self, gt_image, pre_image):
        B,H,W = gt_image.shape
        from tkinter import _flatten
        gt_image_list = np.split(gt_image, B)
        pre_image_list = np.split(pre_image, B)


        for gt_batch, pre_batch in zip(gt_image_list,pre_image_list):
            confusion_matrix = self._generate_matrix(gt_batch, pre_batch)
            Acc = (np.diag(confusion_matrix)+1e-5) / (confusion_matrix.sum(axis=1) + 1e-5)
            Rec = (np.diag(confusion_matrix)+1e-5) / (confusion_matrix.sum(axis=0) + 1e-5)
            self.precision.append(Acc[1])
            self.recall.append(Rec[1])           


    def _add_batch(self, gt_image, pre_image):
        B,H,W = gt_image.shape
        from tkinter import _flatten
        gt_image_list = np.split(gt_image, B)
        gt_image_list = [np.split(batch, H//32, axis = 1) for batch in gt_image_list]
        gt_image_list = _flatten(gt_image_list)
        gt_image_list = [np.split(batch, W//32, axis = 2) for batch in gt_image_list]
        gt_image_list = _flatten(gt_image_list)


        pre_image_list = np.split(pre_image, B)
        pre_image_list = [np.split(batch, H//32, axis = 1) for batch in pre_image_list]
        pre_image_list = _flatten(pre_image_list)
        pre_image_list = [np.split(batch, W//32, axis = 2) for batch in pre_image_list]
        pre_image_list = _flatten(pre_image_list)

        for gt_batch, pre_batch in zip(gt_image_list,pre_image_list):
            self._calcu_batch(gt_batch, pre_batch)

    def pr_curve(self):
        # Accuracy = (self.TP+1e-5) / (self.TP + self.FP + 1e-5)
        # Recall = (self.TP+1e-5) / (self.TP + self.FN + 1e-5)
        # return Accuracy, Recall
        print('precision',self.precision)
        print('recall',self.recall)
        return np.array(self.precision),np.array(self.recall)


    def _calcu_batch(self, gt_image_batch, pre_image_batch):
        mask = (gt_image_batch >= 0) & (gt_image_batch < self.num_class)
        label = self.num_class * gt_image_batch[mask].astype('int') + pre_image_batch[mask]#真值*类别数 + 预测值
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)

        CIoU = (np.diag(confusion_matrix) +1e-5)/ (
                    np.sum(confusion_matrix, axis=1) + 
                    np.sum(confusion_matrix, axis=0) -
                    np.diag(confusion_matrix) + 1e-5)
        # CIoU = CIoU[self.num_class-1]
        th = np.linspace(0.5,0.99,50)
        # if gt_image_batch.sum() > 0:   # positive sample
        #     self.TP += (CIoU[1]>=th).astype(np.int)
        #     self.FP += (CIoU[1]<th).astype(np.int)
        # else:                           #negative sample
        #     self.TN += (CIoU[0]>=th).astype(np.int)
        #     self.FN += (CIoU[0]<th).astype(np.int)

        self.TP += (CIoU[1]>=th).astype(np.int)
        self.FP += (CIoU[1]<th).astype(np.int)
        self.TN += (CIoU[0]>=th).astype(np.int)
        self.FN += (CIoU[0]<th).astype(np.int)

        


        return confusion_matrix
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        self.precision = []
        self.recall = []

    def result(self, types=None):
        if types is None:
            Acc = self.Pixel_Accuracy()
            Acc_class = self.Pixel_Accuracy_Class()
            Rec_class = self.Pixel_Recall_class()
            mIoU = self.Mean_Intersection_over_Union()
            Fscore = self.FScore()
            FWIoU = self.Frequency_Weighted_Intersection_over_Union()
            return Acc,Acc_class,Rec_class,mIoU,FWIoU,Fscore
        else:
            return None




