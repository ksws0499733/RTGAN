
import os

from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

from dataloaders import make_data_loader
from modeling import build_generator
import torch.nn as nn

NUM_CLASSES = 2
class Base_trainer(object):

    def __init__(self, args):
        self.args = args

        #1----- Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        self.args.outputFile = os.path.join(self.saver.experiment_dir,'output')
        if not os.path.exists(self.args.outputFile):
                os.makedirs(self.args.outputFile)

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        #2----- Define Dataloader

        data_loader_out = make_data_loader(args)

        self.train_loader = data_loader_out[0]
        self.val_loader   = data_loader_out[1]
        self.test_loader  = data_loader_out[2]
        self.nclass       = data_loader_out[3]
        #3----- Define GAN model
        
        #4------ Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        self.evaluator_mask = Evaluator(self.nclass)
        self.evaluator_nomask = Evaluator(self.nclass)

        #7------ 恢复模型（检查点） Resuming checkpoint
        self.best_pred = 0.0
        if args.ft:
            args.start_epoch = 0

    def add_evaluatorScalar(self, evaluator, head='', epoch = 0):
        Acc,Acc_class,Rec_class,mIoU,FWIoU,Fscore = evaluator.result()
        # self.writer.add_scalar('val/{}Acc'.format(head), Acc, epoch)
        for iid in range(self.nclass):
            # self.writer.add_scalar('val/fwIoU{}'.format(iid), FWIoU[iid], epoch)
            self.writer.add_scalar('val/{}IoU_cls{}'.format(head,iid), 
                                                        mIoU[iid], 
                                                        epoch)
            self.writer.add_scalar('val/{}Acc_cls{}'.format(head,iid), 
                                                            Acc_class[iid], 
                                                            epoch)
    def _load_Model(self, args, num_classes, in_chans, resume):
        
        model = build_generator(args.backbone,
                                neck=args.neck,
                                num_classes=num_classes,
                                in_chans=in_chans,
                                pretrain=False)      
        if args.cuda:
            model = nn.DataParallel(model).cuda()
        if resume is not None:

            model.load_state_dict(resume,strict=False)

        return model