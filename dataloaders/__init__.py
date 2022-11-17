
import torch.utils.data as tdata
from torch.utils.data import DataLoader
from dataloaders.utils import collate_fn


def make_data_loader(args):

    print('\n------make_data_loader------\n')

    if args.dataset == 'iRailway':
        from .datasets.iRailwayA import buildA as iRaily_build

        num_class = 2
        print('  dataset: iRailway0')
        print('  num_class:',num_class)
        dataset_train = iRaily_build(image_set='iRailway',split='train', args=args)
        dataset_val = iRaily_build(image_set='iRailway',split='val', args=args)
        dataset_test = iRaily_build(image_set='iRailway',split='test', args=args)

        print('  train samples:',len(dataset_train))
        print('  val   samples:',len(dataset_val))
        print('  test  samples:',len(dataset_test))

        train_loader = DataLoader(dataset_train, 
                                    batch_size=args.batch_size, 
                                    shuffle=True,
                                    collate_fn=collate_fn)
        val_loader = DataLoader(dataset_val, 
                                    batch_size=args.batch_size, 
                                    shuffle=False,
                                    collate_fn=collate_fn)
        test_loader = DataLoader(dataset_test, 
                                    batch_size=args.batch_size, 
                                    shuffle=False,
                                    collate_fn=collate_fn)

        return train_loader, val_loader, test_loader, num_class
    else:
        raise NotImplementedError

