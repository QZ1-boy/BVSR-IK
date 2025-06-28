import os
import configparser

class Config_KCA:
    def __init__(self, config_path):
        parser = configparser.ConfigParser()
        parser.read(config_path)
        # experiment
        self.seed = int(parser.get('experiment', 'seed'))
        # training
        self.dataset_path = parser.get('training', 'dataset_path')
        self.anno_path = parser.get('training', 'anno_path')
        self.test_dataset_path = parser.get('test', 'test_dataset_path')
        self.save_dir = parser.get('training', 'save_dir')
        self.log_dir = parser.get('training', 'log_dir')
        self.log_dir = os.path.join(self.save_dir, self.log_dir)
        
        self.nThreads = int(parser.get("training", "nThreads"))
        self.num_epochs = int(parser.get("training", "num_epochs"))
        self.R_lr = float(parser.get("training", "R_lr"))
        self.batch_size = int(parser.get('training', 'batch_size'))
        self.patch_size = int(parser.get('training', 'patch_size'))
        self.finetuning = (parser.get('training', 'finetuning') == 'True')

        self.scale = int(parser.get('training', 'scale'))
        self.num_seq = int(parser.get('training', 'num_seq'))

        self.corrected_loss_weight = float(parser.get("training", "corrected_loss_weight"))

        self.gpu = parser.get("training", "gpu")

        # Network
        self.in_channels = int(parser.get('network', 'in_channels'))
        self.dim = int(parser.get('network', 'dim'))
        self.Knum = int(parser.get('network', 'Knum'))
        self.ds_kernel_size = int(parser.get('network', 'ds_kernel_size'))
        self.us_kernel_size = int(parser.get('network', 'us_kernel_size'))
        self.bias = (parser.get('network', 'bias') == 'True')

        # validation
        self.val_period = int(parser.get('validation', 'val_period'))

        # test
        self.custom_path = parser.get('test', 'custom_path')
        self.stage_ckp_path = parser.get('test', 'stage_ckp_path')
        self.test_epoch = int(parser.get("test", "test_epoch"))

