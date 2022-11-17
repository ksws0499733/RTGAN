
import imp
import os
import numpy as np
from tqdm import tqdm

from utils.loss import SegmentationLosses, GradientLosses
from utils.lr_scheduler import LR_Scheduler
from modeling import build_generator
import torch
from .saveimage import saveimage
from .base_trainer import Base_trainer

import matplotlib.pyplot as plt

NUM_CLASSES = 2
class Model_Trainer(Base_trainer):

    def __init__(self, args):
        super(Model_Trainer, self).__init__(args)
        self.args = args

        if args.calcu_flops:
            self._calcu_flops(args)

        #3----- Define GAN model
        self.model = build_generator(args.backbone,
                                neck=args.neck,
                                num_classes=NUM_CLASSES,
                                in_chans=3,
                                pretrain=False)
        


        #4-----Define train params
        print('\n------ Other Paras-------\n')
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                    lr=args.lr,
                                    betas=(0.9,0.999),
                                    weight_decay=args.weight_decay)

        self.optimizer = optimizer
        #6----- Define Loss function
        #6.1----- whether to use class balanced weights
        if args.use_balanced_weights:
            weight = torch.from_numpy(np.array(1, 40).astype(np.float32))
            print('weight= ',weight)
        else:
            weight = torch.from_numpy(np.array([1, 10]).astype(np.float32))
        # weight = None
        #6.2------ Loss function
        criterion = SegmentationLosses(weight=weight, 
                                cuda=args.cuda).build_loss(mode=args.loss_type)
        criterion2 = GradientLosses(cuda=args.cuda,
                                gauss_kernel_size = 17,
                                gauss_kernel_sigma = 2.0).build_loss(mode="rawInput")
        self.criterion = criterion
        self.criterion_g = criterion2

        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                     args.epochs, len(self.train_loader))

        if args.cuda:     
            print('\t---cuda----')       
            self.model = torch.nn.DataParallel(self.model.cuda(), device_ids=[0,1])
        
        if args.resume is not None:
            #9.1--- load checkpoint
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            print(checkpoint.keys())
            print(len(checkpoint['generator']))

            self.model.load_state_dict(checkpoint['generator'])

        self.Precision= []
        self.Recall = []
        print('\n------ Start -------\n')

        # print(self.model.backbone)

    def training(self, epoch):
        train_loss = 0.0
        mask_loss = 0.0
        nomask_loss = 0.0
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        tbar.set_description('train Epoch: %d' % (epoch))
        self.scheduler(self.optimizer, 0, epoch, self.best_pred)
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
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()

            output = self.model(image)

            # gt_d = target.clone()
            # gt_d[mask<0.5] = 255
            # mask_loss_itr = self.criterion(output['cls'], gt_d)

            # gt_n = target.clone()
            # gt_n[mask>0.5] = 255
            # nomask_loss_itr = self.criterion(output['cls'], gt_n)

            grad_loss_itr = self.criterion_g(output['cls'], target)
            loss_itr = self.criterion(output['cls'], target)
            side_loss_itr = 0
            for _sideout in output['side']:
                if _sideout is not None:
                    side_loss_itr = side_loss_itr + self.criterion(_sideout, target)

            loss = loss_itr*3 + grad_loss_itr*2 + side_loss_itr*1
            # loss = eval(self.args.loss_msg)


            if torch.isnan(output['cls']).any():
                print('nan')

            loss.backward()
            self.optimizer.step()                   #update model

            #3-------record
            train_loss += loss.item() #record loss
            # mask_loss += mask_loss_itr.item()
            # nomask_loss += nomask_loss_itr.item()
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
        
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        # self.writer.add_scalar('train/mask_loss_epoch', mask_loss, epoch)
        # self.writer.add_scalar('train/nomask_loss_epoch', nomask_loss, epoch)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        self.evaluator_mask.reset()
        self.evaluator_nomask.reset()
        tbar = tqdm(self.val_loader, desc='\r')

        tbar.set_description('valid Epoch: %d' % (epoch))
        for i, sample in enumerate(tbar):
            
            #1-----sample
            image= sample['image']
            target = sample['label']
            mask = sample['mask']
            
            if self.args.cuda:
                image= image.cuda()
                target = target.cuda()
                mask = mask.cuda()

            with torch.no_grad():
                outputs = self.model(image)['cls']

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

                
        with open(os.path.join(self.args.outputFile, 'epoch_result.txt'), 'a') as f:
            f.write('Validation:\n')
            f.write('[Epoch: %d, numImages: %5d]\n' % (epoch, i * self.args.batch_size + image.data.shape[0]))
            f.write("Acc:{}\n Acc_class:{}\n Rec_class:{}\n".format(Acc, Acc_class, Rec_class))            
            f.write("mIoU:{}\n FWIoU:{}\n Fscore:{}\n".format(mIoU, FWIoU, Fscore))            
            f.write('IOU: %.3f\n\n' % mIoU.mean())

            
        new_pred = Fscore[1]
        new_iou = mIoU[1]
        new_recall = self.evaluator.Pixel_Recall_class()[1]
        new_precision = self.evaluator.Pixel_Accuracy_Class()[1]

        if new_pred > self.best_pred:
            print("****save model***")
            torch.save({
                        'epoch': epoch,
                        'generator': self.model.state_dict()
                         }, 
                    os.path.join(self.saver.experiment_dir,'best_model.pth'))

            self.best_pred = new_pred
            with open(os.path.join(self.args.outputFile, 'best_pred.txt'), 'a') as f:
                f.write("Epoch: %d, F:%f, iou:%f, R:%f, P:%f\n"%(epoch, 
                                                                new_pred, 
                                                                new_iou,
                                                                new_recall,
                                                                new_precision))

    def test(self, epoch):
        print(self.args.outputFile)

        self.model.eval()
        if epoch > 0:
            tbar = tqdm(self.val_loader, desc='\r')
        else:
            tbar = tqdm(self.test_loader, desc='\r')
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
                outputs = self.model(image)['cls']

            pred = outputs.data.cpu().numpy() 
            # pred = np.argmax(pred, axis=1)      # pred mask
            gt = target.data.cpu().numpy()
            saveimage(image.cpu().permute(0,2,3,1).numpy() ,
                                    pred,
                                    root=self.args.outputFile,
                                    subroot='ep%03d'%epoch,
                                    startID=i*self.args.batch_size, 
                                    gt=gt)


    def _calcu_flops(self, args):
        model = build_generator(args.backbone,
                                neck=args.neck,
                                num_classes=NUM_CLASSES,
                                in_chans=3,
                                pretrain=False)

        from thop import profile
        input_test = torch.randn(1,3,512,512)
        if args.cuda:     
            input_test = input_test.cuda()
        
        flops, params = profile(model, inputs=(input_test,))
        model_name = args.backbone+'-'+args.neck
        print("%s | %.3f | %.3f"%(model_name, params/(1000**2), flops/(1000**3)))

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # GPU预热
        for _ in range(50):
            _ = model(input_test)

        # 测速
        iterations = 100
        times = torch.zeros(iterations)     # 存储每轮iteration的时间
        with torch.no_grad():
            for iter in range(iterations):
                starter.record()
                _ = model(input_test)
                ender.record()
                # 同步GPU时间
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender) # 计算时间
                times[iter] = curr_time
                # print(curr_time)

        mean_time = times.mean().item()
        print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))

