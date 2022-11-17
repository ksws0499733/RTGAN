class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'iRailway':
            # dataset_root = r'/home/user1106/DataSet/all_dataset2'
            # cfg_file = r'/home/user1106/DataSet/all_dataset2/dataset_used.txt'
            dataset_root = r'doc/'
            cfg_file = r'doc/dataset_used.txt'
            return dataset_root, cfg_file
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
