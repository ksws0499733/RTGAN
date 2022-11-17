
import os
import numpy as np
from tqdm import tqdm

from modeling import build_generator
import torch.nn as nn
from gan.models import RailModel
import torch
from .saveimage import saveimage
from .base_trainer import Base_trainer


NUM_CLASSES = 2
class GAN_Trainer(Base_trainer):

    def __init__(self, args):
        super(GAN_Trainer, self).__init__(args)
        self.args = args

        #3----- Define GAN model
        generator = build_generator(args.backbone,
                                neck=args.neck,
                                num_classes=NUM_CLASSES,
                                in_chans=3,
                                pretrain=False)

        print("-----resume file-----:",args.generator_model)
        dic = torch.load(args.generator_model, map_location='cuda')

        self.model = RailModel(args.CFG,generator,dic,numclass=NUM_CLASSES)



    def training(self, epoch):

        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        tbar.set_description('train Epoch: %d' % (epoch))
        for i, sample in enumerate(tbar):

            self.model.train()

            #1-----sample
            image= sample['image']
            target = sample['label']
            mask = sample['mask']
            
            if self.args.cuda:
                image= image.cuda()
                target = target.cuda()
                mask = mask.cuda()

            outputs, gen_loss, dis_loss, logs = self.model.process(image, target)

            if i%5 in [0,1,2,3]:
                # backward
                self.model.backward(None, dis_loss)
            else:
                self.model.backward(gen_loss, None)

            self.writer.add_scalar('train/gen_loss_iter', gen_loss.item(), i + num_img_tr * epoch)
            self.writer.add_scalar('train/dis_loss_iter', dis_loss.item(), i + num_img_tr * epoch)
        
    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        self.evaluator_mask.reset()
        self.evaluator_nomask.reset()
        tbar = tqdm(self.val_loader, desc='\r')

        tbar.set_description('valid Epoch: %d' % (epoch))
        for i, sample in enumerate(tbar):
            
            #1-----sample
            image= sample['image'] #一个字典，字典包含{image,label}两个key
            target = sample['label']  #一个字典，字典包含{image,label}两个key
            mask = sample['mask']
            
            if self.args.cuda:
                image= image.cuda()
                target = target.cuda()
                mask = mask.cuda()

            with torch.no_grad():
                outputs, gen_loss, dis_loss, logs = self.model.process(image, target)

                pred = outputs.data.cpu().numpy() 
                pred = np.argmax(pred, axis=1)      # pred mask
                
                gt = target.clone().cpu().numpy()       # ground truth
                self.evaluator.add_batch(gt, pred)

                mask = mask.data.cpu().numpy() 
                gt_d = target.clone().cpu().numpy()
                gt_d[mask<0.5] = 255
                self.evaluator_mask.add_batch(gt_d, pred)

                gt_n = target.clone().cpu().numpy()
                gt_n[mask>0.5] = 255
                self.evaluator_nomask.add_batch(gt_n, pred)

        self.add_evaluatorScalar(self.evaluator,'total',epoch)
        self.add_evaluatorScalar(self.evaluator_mask,'mask',epoch)
        self.add_evaluatorScalar(self.evaluator_nomask,'nomask',epoch)
        
        Acc,Acc_class,Rec_class,mIoU,FWIoU,Fscore = self.evaluator.result()
        Acc2,Acc_class2,Rec_class2,mIoU2,FWIoU2,Fscore2 = self.evaluator_nomask.result()
                
        with open(os.path.join(self.args.outputFile, 'epoch_result.txt'), 'a') as f:
            f.write('Validation:\n')

            f.write('[Epoch: %d, numImages: %5d]\n' % (epoch, i * self.args.batch_size + image.data.shape[0]))

            f.write("Acc:{}\n Acc_class:{}\n Rec_class:{}\n".format(Acc, Acc_class, Rec_class)) 

            f.write("mIoU:{}\n FWIoU:{}\n Fscore:{}\n".format(mIoU, FWIoU, Fscore))       

            f.write('IOU: %.3f\n\n' % mIoU.mean())


        with open(os.path.join(self.args.outputFile, 'epoch_result2.txt'), 'a') as f:
            f.write('Validation:\n')

            f.write('[Epoch: %d, numImages: %5d]\n' % (epoch, i * self.args.batch_size + image.data.shape[0]))

            f.write("Acc:{}\n Acc_class:{}\n Rec_class:{}\n".format(Acc2, Acc_class2, Rec_class2)) 

            f.write("mIoU:{}\n FWIoU:{}\n Fscore:{}\n".format(mIoU2, FWIoU2, Fscore2))    

            f.write('IOU: %.3f\n\n' % mIoU2.mean())
        new_pred = Fscore[1]
        new_iou = mIoU[1]
        new_recall = self.evaluator.Pixel_Recall_class()[1]
        new_precision = self.evaluator.Pixel_Accuracy_Class()[1]

        new_pred2 = Fscore2[1]
        new_iou2 = mIoU2[1]
        new_recall2 = self.evaluator_nomask.Pixel_Recall_class()[1]
        new_precision2 = self.evaluator_nomask.Pixel_Accuracy_Class()[1]

        if new_pred > self.best_pred:
            print("****save model***")
            self.model.save(self.saver.experiment_dir)

            self.best_pred = new_pred
            with open(os.path.join(self.args.outputFile, 'best_pred.txt'), 'a') as f:
                f.write("Epoch: %d, F:%f, iou:%f, R:%f, P:%f\n"%(epoch, 
                                                                new_pred, 
                                                                new_iou,
                                                                new_recall,
                                                                new_precision))    
                f.write("\t\t F2:%f, iou2:%f, R2:%f, P2:%f\n"%( new_pred2, 
                                                                new_iou2,
                                                                new_recall2,
                                                                new_precision2))      

    def test(self, epoch):
        print(self.args.outputFile)

        self.model.eval()

        if epoch >= 0:
            tbar = tqdm(self.val_loader, desc='\r')
        else:
            tbar = tqdm(self.test_loader, desc='\r')
        # tbar = tqdm(self.test_loader, desc='\r')
        tbar.set_description('test Epoch: %d' % (epoch))
        for i, sample in enumerate(tbar):
            
            image= sample['image']  
            target = sample['label']  
            mask = sample['mask']
            if self.args.cuda:
                image= image.cuda()
                target = target.cuda()
                mask = mask.cuda()

            with torch.no_grad():
                outputs, gen_loss, dis_loss, logs = self.model.process(image, target)

            pred = outputs.data.cpu().numpy() 
            # pred = np.argmax(pred, axis=1)      # pred mask
            gt = target.data.cpu().numpy()
            saveimage(image.cpu().permute(0,2,3,1).numpy() ,
                                    pred,
                                    root=self.args.outputFile,
                                    subroot='ep%03d'%epoch,
                                    startID=i*self.args.batch_size, 
                                    gt=gt)




