import os
import torch
import random
import argparse
import numpy as np
from model_KCA import KCANet
from train import Trainer_KCA_Gaussian, Trainer_KCA_Realistic
from utils import Report
from data import get_dataset
from config import Config_KCA
import warnings
warnings.filterwarnings("ignore")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(config):
    global_step = 0
    train_log = Report(config.save_dir, type='train', stage=1)
    val_log = Report(config.save_dir, type='val', stage=1)

    train_dataloader = get_dataset(config, type='train')
    valid_dataloader = get_dataset(config, type='val')

    model = KCANet(config=config)
    if config.dataloader == 'REDS_Gaussian':
        trainer = Trainer_KCA_Gaussian(config=config, model=model)
    elif config.dataloader == 'REDS_Realistic':
        trainer = Trainer_KCA_Realistic(config=config, model=model)
    elif config.dataloader == 'Vimeo_Gaussian':
        trainer = Trainer_KCA_Gaussian(config=config, model=model)
    elif config.dataloader == 'Vimeo_Realistic':
        trainer = Trainer_KCA_Realistic(config=config, model=model)
    else:
        trainer = Trainer(config=config, model=model)

    print(f'num parameters: {count_parameters(model)}')


    best_psnr = 0
    last_epoch = 0
    trainer.load_before_checkpoint(R_ckp_path=config.stage_ckp_path)
    if config.finetuning:
        last_epoch = trainer.load_checkpoint()

    torch.autograd.set_detect_anomaly(True)
    for epoch in range(last_epoch, config.num_epochs):
        train_log.write(f'========= Epoch {epoch+1} of {config.num_epochs} =========')
        global_step = trainer.train(train_dataloader, train_log, global_step)

        if (epoch + 1) % config.val_period == 0 or epoch == config.num_epochs - 1:
            psnr = trainer.validate(valid_dataloader, val_log, epoch+1)
            trainer.save_checkpoint(epoch + 1)
            if psnr > best_psnr:
                best_psnr = psnr
                trainer.save_best_model(epoch + 1)


def test(args, config):
    test_dataloader = get_dataset(config, type='test')
    model = KCANet(config=config)
    trainer = Trainer(config=config, model=model)
    save_epoch = trainer.load_best_model()
    trainer.test(test_dataloader)
    trainer.test_quantitative_result(epoch=save_epoch, gt_dir=os.path.join(config.test_dataset_path, 'val_sharp'),
                                     output_dir=config.save_dir, image_border=config.num_seq//2)


def test_custom(args, config):
    from data import Custom_Dataset

    data = Custom_Dataset(config)
    test_dataloader = torch.utils.data.DataLoader(data, batch_size=1, drop_last=False, shuffle=False, num_workers=int(config.nThreads), pin_memory=True)
    model = KCANet(config=config)
    trainer = Trainer(config=config, model=model)
    save_epoch = trainer.load_best_model()
    trainer.test(test_dataloader)
    trainer.test_quantitative_result(epoch=save_epoch, gt_dir=os.path.join(config.custom_path, 'HR'), output_dir=config.save_dir, image_border=config.num_seq//2)


def test_REDS4(args, config, REDS4_path):
    from data import Custom_Dataset

    data = Custom_Dataset(config, REDS4_path)
    # print('REDS4_path',REDS4_path)
    test_dataloader = torch.utils.data.DataLoader(data, batch_size=1, drop_last=False, shuffle=False, num_workers=int(config.nThreads), pin_memory=True)
    model = KCANet(config=config)
    if args.Deg_option == 'Gaussian_REDS':
        trainer = Trainer_KCA_Gaussian(config=config, model=model)
    elif args.Deg_option == 'Realistic_REDS':
        trainer = Trainer_KCA_Realistic(config=config, model=model)
    # save_epoch = trainer.load_best_model()
    save_epoch = trainer.load_checkpoint(epoch=400) 
    trainer.test(test_dataloader)
    
    trainer.test_quantitative_result(epoch=save_epoch,gt_dir=os.path.join(REDS4_path, 'HR'),
                                     output_dir=os.path.join(config.save_dir,config.save_dataset), image_border=config.num_seq//2)
                                     
def test_Vid4(args, config, Vid4_path):
    from data import Custom_Dataset
    # print('Vid4_path',Vid4_path)

    data = Custom_Dataset(config, Vid4_path)
    test_dataloader = torch.utils.data.DataLoader(data, batch_size=1, drop_last=False, shuffle=False, num_workers=int(config.nThreads), pin_memory=True)
    model = KCANet(config=config)
    if args.Deg_option == 'Gaussian_REDS':
        trainer = Trainer_KCA_Gaussian(config=config, model=model)
    elif args.Deg_option == 'Realistic_REDS':
        trainer = Trainer_KCA_Realistic(config=config, model=model)
    # save_epoch = trainer.load_best_model()
    save_epoch = trainer.load_checkpoint(epoch=392)
    trainer.test(test_dataloader)
    trainer.test_quantitative_result(epoch=save_epoch, gt_dir=os.path.join(Vid4_path, 'HR'),
                                     output_dir=os.path.join(config.save_dir,config.save_dataset), image_border=config.num_seq//2)


def test_UDM10(args, config, UDM10_path):
    from data import Custom_Dataset

    data = Custom_Dataset(config, UDM10_path)
    test_dataloader = torch.utils.data.DataLoader(data, batch_size=1, drop_last=False, shuffle=False, num_workers=int(config.nThreads), pin_memory=True)
    model = KCANet(config=config)
    if args.Deg_option == 'Gaussian_REDS':
        trainer = Trainer_KCA_Gaussian(config=config, model=model)
    elif args.Deg_option == 'Realistic_REDS':
        trainer = Trainer_KCA_Realistic(config=config, model=model)
    # save_epoch = trainer.load_best_model()
    save_epoch = trainer.load_checkpoint(epoch=396)  
    trainer.test(test_dataloader)
    trainer.test_quantitative_result(epoch=save_epoch, gt_dir=os.path.join(UDM10_path, 'HR'),
                                     output_dir=os.path.join(config.save_dir,config.save_dataset), image_border=config.num_seq//2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train KCANet on REDS')
    parser.add_argument('--test', action='store_true', help='test KCANet on REDS4')
    parser.add_argument('--test_custom', action='store_true', help='test KCANet on custom dataset')
    parser.add_argument('--test_REDS4', action='store_true', help='test KCANet on REDS4 dataset')
    parser.add_argument('--test_Vid4', action='store_true', help='test KCANet on Vid4 dataset')
    parser.add_argument('--test_UDM10', action='store_true', help='test KCANet on SPMCS dataset')
    parser.add_argument('--config_path', type=str, default='./experiment.cfg', help='path to config file with hyperparameters, etc.')
    parser.add_argument('--Deg_option', type=str, default='Gaussian', help='path to config file with hyperparameters, etc.')

    args = parser.parse_args()

    config = Config_KCA(args.config_path)
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if args.train:
        if args.Deg_option == 'Gaussian_REDS':
            config.save_dataset = 'REDS4_Gaussian'
            config.dataloader = 'REDS_Gaussian'
            train(config)
        elif args.Deg_option == 'Realistic_REDS':
            config.save_dataset = 'REDS4_Realistic'
            config.dataloader = 'REDS_Realistic'
            train(config)
        else:
            config.save_dataset = 'REDS4_org'
            train(config)

    if args.test:
        config.save_dataset = 'REDS4_org'
        test(config)


    if args.test_REDS4:
        if args.Deg_option == 'Gaussian_REDS':
            config.save_dataset = 'REDS4_Gaussian'
            REDS4_path = './dataset/REDS4_BlurDown_Gaussian'
            test_REDS4(args, config, REDS4_path)
        elif args.Deg_option == 'Realistic_REDS':
            config.save_dataset = 'REDS4_Realistic'
            REDS4_path = './dataset/REDS4_BlurDown_Realistic'
            test_REDS4(args, config, REDS4_path)

    if args.test_Vid4:
        if args.Deg_option == 'Gaussian_REDS':
            config.save_dataset = 'Vid4_Gaussian'
            Vid4_path = './dataset/Vid4_BlurDown_Gaussian'
            test_Vid4(args, config, Vid4_path)
        elif args.Deg_option == 'Realistic_REDS':
            config.save_dataset = 'Vid4_Realistic'
            Vid4_path = './dataset/Vid4_BlurDown_Realistic'
            test_Vid4(args, config, Vid4_path)

    if args.test_UDM10:
        if args.Deg_option == 'Gaussian_REDS':
            config.save_dataset = 'UDM10_Gaussian'
            UDM10_path = './dataset/UDM10_BlurDown_Gaussian'
            test_UDM10(args, config, UDM10_path)
        elif args.Deg_option == 'Realistic_REDS':
            config.save_dataset = 'UDM10_Realistic'
            UDM10_path = './dataset/UDM10_BlurDown_Realistic'
            test_UDM10(args, config, UDM10_path)

  
    
    