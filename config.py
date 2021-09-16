from easydict import EasyDict as edict
import os

config = edict()
config.cuda_devices = [0, 1, 2, 3]
config.pretrained = False
config.pretrained_model = '/path/to/pretrained/model.pth.tar'
config.nEpochs = 400
config.resume_epoch = 0  # Default is 0, change if want to resume
config.useTest = False # See evolution of the test set when training
config.nTestInterval = 10 # Run on test set every nTestInterval epochs
config.snapshot = 100 # Store a model every snapshot epochs
config.num_workers = 8 # num of worker for dataloader

config.batch_size = 1

config.attention = '' # 'spatio_temporal', 'channel'
config.optimizer = 'adam'
config.lr = 5e-5 # Learning rate ## default 1e-3
config.lr_step = 25
config.gamma = 0.9
config.momentum = 0.9
config.wd = 0 # default 5e-3
config.eps = 1e-8 # eps for adam optimizer

config.dataset = 'ucf101'
config.sfe_type = 't1'
config.ar3d_version = 'v2'
config.frames_per_clips = 16
config.hidden_units = 4096
config.reduction_ratio = 4

if config.dataset == 'hmdb51':
    config.num_classes=51
    config.labels_path = '/home/subin/AR3D/C3D/dataloaders/hmdb_labels.txt'
    config.logging_step = 10
elif config.dataset == 'ucf101':
    config.num_classes = 101
    config.labels_path = '/home/subin/AR3D/C3D/dataloaders/ucf_labels.txt'
    config.logging_step = 30
elif config.dataset == 'kinetics400':
    config.num_classes = 400
    config.labels_path = '/home/subin/AR3D/AR3D/dataloaders/kinetics400_lables.txt'
    config.logging_step = 100
else:
    print('We only implemented hmdb, ucf, kinetics400 datasets.')
    raise NotImplementedError

config.save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
config.train_log = './spatio_temporal_adam_hmdb51_5e5_09step1.log'
config.model_name = 'AR3D'


config.frames_per_clips = 16

class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = '/mnt/hdd_10tb/dataset/UCF/UCF_split/train'

            # Save preprocess data into output_dir
            output_dir = '/Users/dj/Downloads/Handover/ucf101_official_split_fold1'

        elif database == 'hmdb51':
            root_dir = '/mnt/hdd_10tb_2/subin/Datasets/video/hmdb51/videos'
            output_dir = '/home/subin/Datasets/hmdb51_split'
            
        elif database == 'kinetics400':
            root_dir = '/mnt/hdd_10tb/dataset/kinetics_400'
            output_dir = '/home/subin/Datasets/kinetics_400'
            # output_dir = '/mnt/usb/dataset/'
            
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

        return root_dir, output_dir
        
    @staticmethod
    def model_dir():
        return 'models/c3d-pretrained.pth'
        # return '/home/subin/AR3D/C3D/run/run_7/models/C3D-ucf101_epoch-49.pth'