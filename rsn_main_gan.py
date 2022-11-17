import argparse
import os
import torch

def main():
    parser = argparse.ArgumentParser(description="RT-GAN Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        help='backbone name (default: resnet)')

    parser.add_argument('--neck', type=str, default='LSN',
                        help='neck name (default: LSN)')

    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')


    # data set
    parser.add_argument('--dataset', type=str, default='iRailwayA',
                        choices=['iRailway','iRailway0','iRailwayA'],
                        help='dataset name (default: iRailway)')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--dataAug', type=str, default='no',
                        help='data argument type(default: all)')
    parser.add_argument('--testPath', type=str, default=r'doc/test_data')

    parser.add_argument('--dataRepeat', type=int, default=10)

    parser.add_argument('--add-opp', action='store_true', default=True,
                        help='add opp sample')

    # train set
    parser.add_argument('--trainRepeat', type=int, default=1,
                        help='number of train Repeat times to train (default: 1)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=16,
                        metavar='N', help='input batch size for training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    parser.add_argument('--test-only', action='store_true', default=False,
                        help='only test (default: False)')
    parser.add_argument('--calcu-flops', action='store_true', default=False,
                        help='calculate model para numbers and flops')

    # train optimizer 
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')

    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')

    parser.add_argument('--generator-model', type=str, default='basemodel/gnn/model_para.pkl')                    
    
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=5,
                        help='evaluuation interval (default: 10)')
    parser.add_argument('--test-interval', type=int, default=10,
                        help='test interval (default: 10)')

    # loss type
    parser.add_argument('--loss-type', type=str, default='hinge',
                        choices=['hinge','bce', 'mse'],
                        help='loss func type (default: hinge)')
    parser.add_argument('--loss-msg', type=str, 
                        default='no')    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
            
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        args.lr = 0.01 / (4 * len(args.gpu_ids)) / args.batch_size

    from gan.config import DEFAULT_CONFIG, Config

    
    torch.manual_seed(args.seed)

    print('\n-------- args ------\n')
    print('  cuda:',args.cuda)
    print('  Starting Epoch:', args.start_epoch)
    print('  Total Epoches:', args.epochs)

    backbones = args.backbone.split(',')
    necks = args.neck.split(',')

    print('  bbones:',len(backbones)," :",backbones)
    print('  necks:',len(necks)," :",necks)

    for bb in backbones:
        for nk in necks:            
            args.backbone = bb
            args.neck = nk            
            args.checkname = 'GAN-'+args.backbone+'-'+args.neck+ \
                        '-'+args.loss_type+'-' +\
                        args.dataset+'-'+ \
                        "dr%d"%args.dataRepeat

            CFG = DEFAULT_CONFIG
            CFG['GPU']=[0,1]
            CFG['PATH']=os.path.join('run',args.dataset, args.checkname)

            CFG['GAN_LOSS'] = args.loss_type
            args.CFG = Config(CFG)

            for itr in range(args.trainRepeat): 
                print('\n-------- train itr %d ------\n'%itr)
                print('    ',bb,' - ',nk)

                from trainer.gan_trainer import GAN_Trainer as Trainer
                trainer = Trainer(args)

                for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
                    if  epoch % 5 == 2:
                        trainer.validation(epoch)  #验证模型
                    if  epoch % 5 == 0:
                        trainer.test(epoch)  #验证模型
                    trainer.training(epoch)
                trainer.test(-1)

                trainer.writer.close()


if __name__ == "__main__":

    
    main()

