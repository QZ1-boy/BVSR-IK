import os
import time
import torch
from datetime import datetime
from utils import Train_KCAReport, TestReport, SaveManager
from thop import profile
# import scipy.io as sio
import cv2
import sys
import math
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import CosineAnnealingRestartLR, CharbonnierLoss
import torch.nn.functional as F



class Trainer_KCA_Gaussian:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        # self.criterion = torch.nn.L1Loss() # MSELoss()
        # self.criterion = torch.nn.MSELoss()
        self.criterion = CharbonnierLoss()
        self.optimizer_R = torch.optim.Adam(self.model.parameters(), lr=self.config.R_lr, betas=[0.9, 0.999],eps=1e-08)
        self.scheduler_R = CosineAnnealingRestartLR(self.optimizer_R, periods=[50, 50, 50, 100], restart_weights=[1,1,1,1], eta_min=1e-7)

        # os.path.dirname
        self.checkpoint_path = os.path.join(self.config.save_dir, f'model_stage')
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        self.model.cuda()

    def save_checkpoint(self, epoch):
        R_state_dict = {'epoch': epoch,
                        'model_R_state_dict': self.model.state_dict(),
                        'optimizer_R_state_dict': self.optimizer_R.state_dict(),
                        'scheduler_R_state_dict': self.scheduler_R.state_dict()}
        torch.save(R_state_dict, self.checkpoint_path + '/model_R_latest.pt')
        torch.save(R_state_dict, self.checkpoint_path + '/model_R_' + str(epoch) + '.pt')

    def save_best_model(self, epoch):
        R_state_dict = {'epoch': epoch,
                        'model_R_state_dict': self.model.state_dict(),
                        'optimizer_R_state_dict': self.optimizer_R.state_dict(),
                        'scheduler_R_state_dict': self.scheduler_R.state_dict()}
        torch.save(R_state_dict, self.checkpoint_path + '/model_R_best.pt')

    def load_checkpoint(self, epoch=None):
        if epoch is None:
            R_state_dict = torch.load(self.checkpoint_path + '/model_R_latest.pt')  # model_R_latest
            self.model.load_state_dict(R_state_dict['model_R_state_dict'])
            self.optimizer_R.load_state_dict(R_state_dict['optimizer_R_state_dict'])
            self.scheduler_R.load_state_dict(R_state_dict['scheduler_R_state_dict'])
            last_epoch = R_state_dict['epoch']
            print(f'load restoration network status from {self.checkpoint_path}/model_R_latest.pt, epoch: {last_epoch}')  # model_R_latest
            print('load_checkpoint [1]')

        else:
            R_state_dict = torch.load(self.checkpoint_path + '/model_R_' + str(epoch) + '.pt')
            self.model.load_state_dict(R_state_dict['model_R_state_dict'])
            self.optimizer_R.load_state_dict(R_state_dict['optimizer_R_state_dict'])
            self.scheduler_R.load_state_dict(R_state_dict['scheduler_R_state_dict'])
            last_epoch = R_state_dict['epoch']
            print(f'load restoration network status from {self.checkpoint_path}/model_R_{epoch}.pt, epoch: {last_epoch}')
            R_state_dict['epoch'] = 396
            torch.save(R_state_dict, self.checkpoint_path + '/model_R_UDM10_396' + '.pt')
            print('Save_checkpoint [2]')

        return last_epoch

    def load_before_checkpoint(self, R_ckp_path):
        R_state_dict = torch.load(R_ckp_path)  # model_R_latest
        self.model.load_state_dict(R_state_dict['model_R_state_dict'])  # , strict=False
        print(f'load restoration network status from {R_ckp_path}')  # model_R_latest

    def load_best_model(self):
        R_state_dict = torch.load(self.checkpoint_path + '/model_R_best.pt')
        self.model.load_state_dict(R_state_dict['model_R_state_dict'])
        print(f'load restoration network status from {self.checkpoint_path}/model_R.pt, epoch: {R_state_dict["epoch"]}')
        save_epoch = R_state_dict["epoch"]

        return save_epoch


    def train(self, dataloader, train_log, global_step):
        self.model.train()
        report = Train_KCAReport()
        start = time.time()

        for idx, (lr_blur_seq, hr_sharp_seq, lr_sharp_seq) in enumerate(dataloader):
            lr_blur_seq = lr_blur_seq.cuda()
            hr_sharp_seq = hr_sharp_seq.cuda()
            lr_sharp_seq = lr_sharp_seq.cuda()

            result_dict = self.model(lr_blur_seq)

            batch_size,  t, _, _, _ = lr_blur_seq.shape    
            restoration_loss = self.criterion(result_dict['outputs'], hr_sharp_seq)
            # 
            # lr_sharp_seq_1 = F.interpolate(lr_sharp_seq, scale_factor=(1,0.5,0.5), mode='trilinear')
            # lr_sharp_seq_2 = F.interpolate(lr_sharp_seq, scale_factor=(1, 0.25,0.25), mode='trilinear')
            # print('lr_sharp_seq',lr_sharp_seq.shape, lr_sharp_seq_1.shape, lr_sharp_seq_2.shape)
            corrected_loss = self.config.corrected_loss_weight * (self.criterion(result_dict['lr_corrs'], lr_sharp_seq) )
                            
            total_loss = restoration_loss + corrected_loss

            self.optimizer_R.zero_grad()
            total_loss.backward()
            self.optimizer_R.step()

            report.update(batch_size, restoration_loss.item(), corrected_loss.item(), total_loss.item())

            global_step += 1

            if global_step % 2000 == 0 or idx == len(dataloader) - 1:
                # if global_step % len(dataloader) == 0: 
                lr_R = self.scheduler_R.optimizer.state_dict()['param_groups'][0]['lr']

                period_time = time.time() - start
                now_time = f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}]-'
                prefix_str = f'[{global_step}/{len(dataloader) * self.config.num_epochs}]  '  #  * self.config.num_epochs
                result_str = report.result_str(lr_R, period_time)

                train_log.write(now_time + prefix_str + result_str)
                start = time.time()
                report.__init__()


        self.scheduler_R.step()

        return global_step

    def validate(self, dataloader, val_log, epoch):
        self.model.eval()
        report = Train_KCAReport()
        start = time.time()

        with torch.no_grad():
            for idx, (lr_blur_seq, hr_sharp_seq, lr_sharp_seq) in enumerate(dataloader):
                lr_blur_seq = lr_blur_seq.cuda()
                hr_sharp_seq = hr_sharp_seq.cuda()
                lr_sharp_seq = lr_sharp_seq.cuda()
                # kernel = kernel.cuda()
                # print('kernel',kernel.shape)

                result_dict = self.model(lr_blur_seq)

                batch_size, _, t, _, _ = lr_blur_seq.shape

                restoration_loss = self.criterion(result_dict['outputs'], hr_sharp_seq)
                # lr_sharp_seq_1 = F.interpolate(lr_sharp_seq, scale_factor=(1,0.5,0.5), mode='trilinear')
                # lr_sharp_seq_2 = F.interpolate(lr_sharp_seq, scale_factor=(1, 0.25,0.25), mode='trilinear')
                corrected_loss = self.config.corrected_loss_weight * (self.criterion(result_dict['lr_corrs'], lr_sharp_seq) )                             
                total_loss = restoration_loss + corrected_loss
                                    
                report.update(batch_size, restoration_loss.item(), corrected_loss.item(),  total_loss.item())
                report.update_correlation_metric(result_dict['lr_corrs'][:, t//2, :,  :, :], lr_sharp_seq[:, t//2, :,  :, :])
                report.update_restoration_metric(result_dict['outputs'][:, t//2, :,  :, :], hr_sharp_seq[:,  t//2, :, :, :])
            
        
        print('[Model: KCA_Gaussian]')
        period_time = time.time() - start
        prefix_str = f'[{epoch}/{self.config.num_epochs}]'
        result_str = report.val_result_str(period_time)
        val_log.write(prefix_str + result_str)

        return report.psnr

    def test(self, dataloader):
        from utils import denorm
        self.model.eval()
        self.save_manager = SaveManager(self.config)

        with torch.no_grad():
            for idx, (lr_blur_seq, filename) in enumerate(dataloader):
                lr_blur_seq = lr_blur_seq.cuda()

                # modelS_ours = self.model.cuda()
                # inputs = torch.rand((2, 5, 3, 256//4, 256//4)).cuda()
                # modelS_ours = modelS_ours.cuda()
                # inputs = inputs.unsqueeze(0)
                # macs, param = (profile(modelS_ours, inputs))
                # print('[KCDA] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)
                # print('lr_blur_seq',inputs.shape) #   1 3 3  180 320 

                # preprocess_time = time.time()
                # result_dict = self.model(lr_blur_seq)
                # print('time',time.time()-preprocess_time)
                # sys.exit(0)
                # print('shape', lr_blur_seq.shape) #  1 5 3 1024 512 
                # sys.exit(0)

                # b,t,c,h,w = lr_blur_seq.shape
                # output = torch.from_numpy(np.zeros([b, t, c, 4*h, 4*w]))
                # lr_corr = torch.from_numpy(np.zeros([b, t, c, h, w]))
                # lr_blur_1 = lr_blur_seq[:,:,:,:h//2 + 4,:]
                # lr_blur_2 = lr_blur_seq[:,:,:,h//2 - 4:,:]
                # result_dict1 = self.model(lr_blur_1)
                # result_dict2 = self.model(lr_blur_2)
                # output1 = result_dict1['outputs']
                # lr_corr1 = result_dict1['lr_corrs']
                # output2 = result_dict2['outputs']
                # lr_corr2 = result_dict2['lr_corrs']

                              
                result_dict = self.model(lr_blur_seq)
                output = result_dict['outputs']
                lr_corr = result_dict['lr_corrs']

                # output[:,:,:,:2*h,:] = output1[:,:,:,:2*h,:]
                # output[:,:,:,2*h:,:] = output2[:,:,:,16:,:]
                # lr_corr[:,:,:,:h//2,:] = lr_corr1[:,:,:,:h//2,:]
                # lr_corr[:,:,:,h//2:,:] = lr_corr2[:,:,:,4:,:]

                output = output.squeeze(dim=0)  # .cpu().detach().numpy()
                output = denorm(output)
                lr_corr = lr_corr.squeeze(dim=0) #  .cpu().detach().numpy()
                lr_corr = denorm(lr_corr)

                filename = filename[0]
                filepath = os.path.basename(os.path.dirname(filename))
                filename = os.path.basename(filename)
                # print('lr_corr',self.config.save_dir, self.config.save_dataset, filepath, filename)
                filename_o = os.path.join(self.config.save_dir, self.config.save_dataset, 'outputs', filepath, filename)
                filename_c = os.path.join(self.config.save_dir, self.config.save_dataset, 'lr_corrs', filepath, filename)
                self.save_manager.save_image(output, filename_o)
                # print('lr_corr',output.shape, lr_corr.shape, kernel.shape)
                self.save_manager.save_image(lr_corr, filename_c)
                print('Save Image:', idx,'...', end="\r")
                # if idx == 2:
                #     sys.exit(0)

    def test_quantitative_result(self, epoch, gt_dir, output_dir, image_border):
        import cv2
        import glob

        report = TestReport(epoch, output_dir)
        scene_list = sorted(glob.glob(os.path.join(gt_dir, '*')))

        for scene in scene_list:
            scene_name = os.path.basename(scene)
            filelist = sorted(glob.glob(os.path.join(scene, '*.png')))
            report.scene_init(scene_name)
            for filename in filelist[image_border:-image_border]:
                gt_img = cv2.imread(filename)
                output_img = cv2.imread(os.path.join(output_dir, 'outputs',scene_name, os.path.basename(filename)))
                report.update_metric(gt_img, output_img, os.path.basename(filename))
                print('Image PSNR:', filename,'...', end="\r")
            report.scene_del(scene_name)

    def process_kernel(self, kernel):
        mi = torch.min(kernel)
        ma = torch.max(kernel)
        kernel = (kernel - mi) / (ma - mi)
        kernel = torch.cat([kernel, kernel, kernel], dim=1)
        kernel = kernel.mul(255.).clamp(0, 255).round()
        return kernel


class Trainer_KCA_Realistic:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        # self.criterion = torch.nn.L1Loss()
        self.criterion = CharbonnierLoss()
   
        self.optimizer_R = torch.optim.Adam(self.model.parameters(), lr=self.config.R_lr, betas=[0.9, 0.999],eps=1e-08)
        self.scheduler_R = CosineAnnealingRestartLR(self.optimizer_R, periods=[50, 50, 50, 100], restart_weights=[1,1,1,1], eta_min=1e-7)

        # os.path.dirname
        self.checkpoint_path = os.path.join(self.config.save_dir, f'model_stage')
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        self.model.cuda()

    def save_checkpoint(self, epoch):
        R_state_dict = {'epoch': epoch,
                            'model_R_state_dict': self.model.state_dict(),
                            'optimizer_R_state_dict': self.optimizer_R.state_dict(),
                            'scheduler_R_state_dict': self.scheduler_R.state_dict()}
        torch.save(R_state_dict, self.checkpoint_path + '/model_R_latest.pt')
        torch.save(R_state_dict, self.checkpoint_path + '/model_R_' + str(epoch) + '.pt')

    def save_best_model(self, epoch):
        R_state_dict = {'epoch': epoch,
                            'model_R_state_dict': self.model.state_dict(),
                            'optimizer_R_state_dict': self.optimizer_R.state_dict(),
                            'scheduler_R_state_dict': self.scheduler_R.state_dict()}
        torch.save(R_state_dict, self.checkpoint_path + '/model_R_best.pt')

    def load_checkpoint(self, epoch=None):
        if epoch is None:
            R_state_dict = torch.load(self.checkpoint_path + '/model_R_latest.pt')  # model_R_latest
            self.model.load_state_dict(R_state_dict['model_R_state_dict']) # , strict=False
            # self.optimizer_R.load_state_dict(R_state_dict['optimizer_R_state_dict'])
            # self.scheduler_R.load_state_dict(R_state_dict['scheduler_R_state_dict'])
            last_epoch = R_state_dict['epoch']
            print(f'load restoration network status from {self.checkpoint_path}/model_R_latest.pt, epoch: {last_epoch}')  # model_R_latest
            print('load_checkpoint [1]')
        else:
            R_state_dict = torch.load(self.checkpoint_path + '/model_R_' + str(epoch) + '.pt')
            self.model.load_state_dict(R_state_dict['model_R_state_dict'])
            self.optimizer_R.load_state_dict(R_state_dict['optimizer_R_state_dict'])
            self.scheduler_R.load_state_dict(R_state_dict['scheduler_R_state_dict'])
            last_epoch = R_state_dict['epoch']
            print(f'load restoration network status from {self.checkpoint_path}/model_R_{epoch}.pt, epoch: {last_epoch}')
            # R_state_dict['epoch'] = 396
            # torch.save(R_state_dict, self.checkpoint_path + '/model_R_UDM10_396' + '.pt')
            # print('Save_checkpoint [2]')

        return last_epoch

    def load_before_checkpoint(self, R_ckp_path):
        R_state_dict = torch.load(R_ckp_path)  # model_R_latest
        self.model.load_state_dict(R_state_dict['model_R_state_dict'])  #  , strict=False
        print(f'load restoration network status from {R_ckp_path}')  # model_R_latest  

    def load_best_model(self):
        R_state_dict = torch.load(self.checkpoint_path + '/model_R_best.pt')
        self.model.load_state_dict(R_state_dict['model_R_state_dict'])
        print(f'load restoration network status from {self.checkpoint_path}/model_R.pt, epoch: {R_state_dict["epoch"]}')
        save_epoch = R_state_dict["epoch"]

        return save_epoch



    def train(self, dataloader, train_log, global_step):
        self.model.train()
        report = Train_KCAReport()
        start = time.time()

        for idx, (lr_blur_seq, hr_sharp_seq, lr_sharp_seq) in enumerate(dataloader):
            lr_blur_seq = lr_blur_seq.cuda()
            hr_sharp_seq = hr_sharp_seq.cuda()
            lr_sharp_seq = lr_sharp_seq.cuda()

            result_dict = self.model(lr_blur_seq)
            batch_size, _, t, _, _ = lr_blur_seq.shape

            restoration_loss = self.criterion(result_dict['outputs'], hr_sharp_seq)
            corrected_loss = self.config.corrected_loss_weight * self.criterion(result_dict['lr_corrs'], lr_sharp_seq)
                            
            total_loss = restoration_loss + corrected_loss 

            self.optimizer_R.zero_grad()
            total_loss.backward()
            self.optimizer_R.step()

            report.update(batch_size, restoration_loss.item(), corrected_loss.item(),  total_loss.item())

            global_step += 1

            if global_step % 2000 == 0 or idx == len(dataloader) - 1:
                # if global_step % len(dataloader) == 0: 
                lr_R = self.scheduler_R.optimizer.state_dict()['param_groups'][0]['lr']

                period_time = time.time() - start
                now_time = f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}]-'
                prefix_str = f'[{global_step}/{len(dataloader) * self.config.num_epochs}]  '  #  * self.config.num_epochs
                result_str = report.result_str(lr_R, period_time)

                train_log.write(now_time + prefix_str + result_str)
                start = time.time()
                report.__init__()

        self.scheduler_R.step()

        return global_step

    def validate(self, dataloader, val_log, epoch):
        self.model.eval()
        report = Train_KCAReport()
        start = time.time()

        with torch.no_grad():
            for idx, (lr_blur_seq, hr_sharp_seq, lr_sharp_seq) in enumerate(dataloader):
                lr_blur_seq = lr_blur_seq.cuda()
                hr_sharp_seq = hr_sharp_seq.cuda()
                lr_sharp_seq = lr_sharp_seq.cuda()

                result_dict = self.model(lr_blur_seq)

                batch_size, _, t, _, _ = lr_blur_seq.shape

                restoration_loss = self.criterion(result_dict['outputs'], hr_sharp_seq)
                corrected_loss = self.config.corrected_loss_weight * self.criterion(result_dict['lr_corrs'], lr_sharp_seq)                               
                total_loss = restoration_loss + corrected_loss 
                                    
                report.update(batch_size, restoration_loss.item(), corrected_loss.item(), total_loss.item())
                report.update_correlation_metric(result_dict['lr_corrs'][:, t//2, :, :, :], lr_sharp_seq[:,  t//2, :, :, :])
                report.update_restoration_metric(result_dict['outputs'][:, t//2, :,  :, :], hr_sharp_seq[:,  t//2, :, :, :])
               
        
        print('[Model: KCA_Realistic]')
        period_time = time.time() - start
        prefix_str = f'[{epoch}/{self.config.num_epochs}]'
        result_str = report.val_result_str(period_time)

        val_log.write(prefix_str + result_str)

        return report.psnr

    def test(self, dataloader):
        from utils import denorm
        self.model.eval()
        self.save_manager = SaveManager(self.config)

        with torch.no_grad():
            video_psnr = []
            video_ssim = []
            for idx, (lr_blur_seq, filename) in enumerate(dataloader):
                lr_blur_seq = lr_blur_seq.cuda()

                # modelS_ours = self.model.cuda()
                # inputs = torch.rand((2, 3, 5, 256//4, 256//4)).cuda()
                # modelS_ours = modelS_ours.cuda()
                # inputs = inputs.unsqueeze(0)
                # macs, param = (profile(modelS_ours, inputs))
                # print('[KCDA] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)
                # preprocess_time = time.time()
                # result_dict = self.model(lr_blur_seq)
                # print('time',time.time()-preprocess_time)
                # # sys.exit(0)

                result_dict = self.model(lr_blur_seq)
                output = result_dict['outputs']
                lr_corr = result_dict['lr_corrs']
               
                output = output.squeeze(dim=0)
                output_save = denorm(output)
                lr_corr = lr_corr.squeeze(dim=0)
                lr_corr_save = denorm(lr_corr) # tensor2numpy

                filename = filename[0]
                filepath = os.path.basename(os.path.dirname(filename))
                filename = os.path.basename(filename)

                filename_o = os.path.join(self.config.save_dir, self.config.save_dataset, 'outputs', filepath, filename)
                filename_c = os.path.join(self.config.save_dir, self.config.save_dataset, 'lr_corrs', filepath, filename)
                self.save_manager.save_image(output_save, filename_o)
                self.save_manager.save_image(lr_corr_save, filename_c)
                # print('lr_corr',lr_corr.shape, output.shape)
                # for ii in range(len(align_list)):
                #     align_img = align_list[ii].squeeze(dim=0)
                #     align_img = denorm(0.2*align_img + 0.8*lr_corr) 
                #     refine_img = refine_list[ii].squeeze(dim=0)
                #     refine_img = denorm(0.2*refine_img + 0.8*lr_corr) 
                #     # print('flow_list[]',refine_img.shape, flow_list[ii].shape)
                #     flow_img = flow_list[ii].squeeze(dim=0).cpu().detach().numpy()
                #     flow_save = flowbatch_to_image(flow_img, None,convert_to_bgr=False)
                #     filename_iter = str(ii) + '_' + filename
                #     filename_f = os.path.join(self.config.save_dir, self.config.save_dataset, 'flow', filepath, filename_iter)
                #     filename_a = os.path.join(self.config.save_dir, self.config.save_dataset, 'aligned', filepath, filename_iter)
                #     filename_r = os.path.join(self.config.save_dir, self.config.save_dataset, 'refined', filepath, filename_iter)
                #     self.save_manager.save_image(flow_save, filename_f)
                #     self.save_manager.save_image(align_img, filename_a)
                #     self.save_manager.save_image(refine_img, filename_r)
                print('Save Image:', idx,'...', end="\r")
        # print('PSNR Image:', sum(video_psnr)/(idx+1),sum(video_ssim)/(idx+1))

    def test_quantitative_result(self, epoch, gt_dir, output_dir, image_border):
        import cv2
        import glob

        report = TestReport(epoch,output_dir)
        scene_list = sorted(glob.glob(os.path.join(gt_dir, '*')))
        # print('scene_list',scene_list)

        for scene in scene_list:
            scene_name = os.path.basename(scene)
            filelist = sorted(glob.glob(os.path.join(scene, '*.png')))
            report.scene_init(scene_name)
            for filename in filelist[image_border:-image_border]:
                # print('filename',output_dir, filename, scene_name)
                gt_img = cv2.imread(filename)  # lr_corr  output LR_blurdown_x4
                output_img = cv2.imread(os.path.join(output_dir, 'outputs',scene_name, os.path.basename(filename)))
                report.update_metric(gt_img, output_img, os.path.basename(filename))
                print('Image PSNR:', filename,'...', end="\r")
            report.scene_del(scene_name)

    def process_kernel(self, kernel):
        mi = torch.min(kernel)
        ma = torch.max(kernel)
        kernel = (kernel - mi) / (ma - mi)
        kernel = torch.cat([kernel, kernel, kernel], dim=1)
        kernel = kernel.mul(255.).clamp(0, 255).round()
        return kernel

    def tensor2numpy(self, tensor, rgb_range=1.):
        rgb_coefficient = 255 / rgb_range
        img = tensor.mul(rgb_coefficient).clamp(0, 255).round()
        img = img[0].data
        img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
        return img
    
    def get_PSNR_SSIM(self, output, gt, crop_border=0):
        cropped_output = output[crop_border:-crop_border, crop_border:-crop_border, :]
        cropped_GT = gt[crop_border:-crop_border, crop_border:-crop_border, :]
        psnr = self.calc_PSNR(cropped_GT, cropped_output)
        ssim = self.calc_SSIM(cropped_GT, cropped_output)
        return psnr, ssim

    def calc_PSNR(self, img1, img2):
        '''
        img1 and img2 have range [0, 255]
        '''
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))

    def calc_SSIM(self, img1, img2):
        '''calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        '''

        def ssim(img1, img2):
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2

            img1 = img1.astype(np.float64)
            img2 = img2.astype(np.float64)
            kernel = cv2.getGaussianKernel(11, 1.5)
            window = np.outer(kernel, kernel.transpose())

            mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
            mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
            sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
            sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                    (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean()

        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim == 2:
            return ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')





class Trainer:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        if self.config.save_train_img:
            self.save_manager = SaveManager(config)
        self.criterion = torch.nn.L1Loss()

        milestones = [260, 360, 380, 390]
        # optimizer and scheduler for degradation learning network
        self.optimizer_D = torch.optim.Adam(self.model.degradation_learning_network.parameters(), lr=self.config.lr)
        self.scheduler_D = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_D, milestones=milestones, gamma=0.5, last_epoch=-1)

        # optimizer and scheduler for restoration network
        if self.config.stage == 2:
            self.optimizer_R = torch.optim.Adam(self.model.restoration_network.parameters(), lr=self.config.lr)
            self.scheduler_R = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_R, milestones=milestones, gamma=0.5, last_epoch=-1)

        # os.path.dirname
        self.checkpoint_path = os.path.join(self.config.save_dir, f'model_stage{self.config.stage}')
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        self.model.cuda()

    def save_checkpoint(self, epoch):
        D_state_dict = {'epoch': epoch,
                        'model_D_state_dict': self.model.degradation_learning_network.state_dict(),
                        'optimizer_D_state_dict': self.optimizer_D.state_dict(),
                        'scheduler_D_state_dict': self.scheduler_D.state_dict()}
        torch.save(D_state_dict, self.checkpoint_path + '/model_D_latest.pt')
        torch.save(D_state_dict, self.checkpoint_path + '/model_D_' + str(epoch) + '.pt')

        if self.config.stage == 2:
            R_state_dict = {'epoch': epoch,
                            'model_R_state_dict': self.model.restoration_network.state_dict(),
                            'optimizer_R_state_dict': self.optimizer_R.state_dict(),
                            'scheduler_R_state_dict': self.scheduler_R.state_dict()}
            torch.save(R_state_dict, self.checkpoint_path + '/model_R_latest.pt')
            torch.save(R_state_dict, self.checkpoint_path + '/model_R_' + str(epoch) + '.pt')

    def save_best_model(self, epoch):
        D_state_dict = {'epoch': epoch,
                        'model_D_state_dict': self.model.degradation_learning_network.state_dict(),
                        'optimizer_D_state_dict': self.optimizer_D.state_dict(),
                        'scheduler_D_state_dict': self.scheduler_D.state_dict()}
        torch.save(D_state_dict, self.checkpoint_path + '/model_D_best.pt')

        if self.config.stage == 2:
            R_state_dict = {'epoch': epoch,
                            'model_R_state_dict': self.model.restoration_network.state_dict(),
                            'optimizer_R_state_dict': self.optimizer_R.state_dict(),
                            'scheduler_R_state_dict': self.scheduler_R.state_dict()}
            torch.save(R_state_dict, self.checkpoint_path + '/model_R_best.pt')

    def load_checkpoint(self, epoch=None):
        if epoch is None:
            D_state_dict = torch.load(self.checkpoint_path + '/model_D_latest.pt')
            self.model.degradation_learning_network.load_state_dict(D_state_dict['model_D_state_dict'])
            self.optimizer_D.load_state_dict(D_state_dict['optimizer_D_state_dict'])
            self.scheduler_D.load_state_dict(D_state_dict['scheduler_D_state_dict'])
            last_epoch = D_state_dict['epoch']
            print(f'load degradation learning network status from {self.checkpoint_path}/model_D_latest.pt, epoch: {last_epoch}')
            print('bbb')

            if self.config.stage == 2:
                R_state_dict = torch.load(self.checkpoint_path + '/model_R_latest.pt')
                self.model.restoration_network.load_state_dict(R_state_dict['model_R_state_dict'])
                self.optimizer_R.load_state_dict(R_state_dict['optimizer_R_state_dict'])
                self.scheduler_R.load_state_dict(R_state_dict['scheduler_R_state_dict'])
                last_epoch = R_state_dict['epoch']
                print(f'load restoration network status from {self.checkpoint_path}/model_R_latest.pt, epoch: {last_epoch}')

            save_epoch = last_epoch
        else:
            D_state_dict = torch.load(self.checkpoint_path + '/model_D_' + str(epoch) + '.pt')
            self.model.degradation_learning_network.load_state_dict(D_state_dict['model_D_state_dict'])
            self.optimizer_D.load_state_dict(D_state_dict['optimizer_D_state_dict'])
            self.scheduler_D.load_state_dict(D_state_dict['scheduler_D_state_dict'])
            last_epoch = D_state_dict['epoch']
            print(f'load degradation learning network status from {self.checkpoint_path}/model_D_{epoch}.pt, epoch: {last_epoch}')
            

            if self.config.stage == 2:
                R_state_dict = torch.load(self.checkpoint_path + '/model_R_' + str(epoch) + '.pt')
                self.model.restoration_network.load_state_dict(R_state_dict['model_R_state_dict'])
                self.optimizer_R.load_state_dict(R_state_dict['optimizer_R_state_dict'])
                self.scheduler_R.load_state_dict(R_state_dict['scheduler_R_state_dict'])
                last_epoch = R_state_dict['epoch']
                print(f'load restoration network status from {self.checkpoint_path}/model_R_{epoch}.pt, epoch: {last_epoch}')
            
        return last_epoch

    def load_best_model(self):
        D_state_dict = torch.load(self.checkpoint_path + '/model_D_best.pt')
        self.model.degradation_learning_network.load_state_dict(D_state_dict['model_D_state_dict'])
        print(f'load degradation learning network status from {self.checkpoint_path}/model_D_best.pt, epoch: {D_state_dict["epoch"]}')
        save_epoch = D_state_dict["epoch"]

        if self.config.stage == 2:
            R_state_dict = torch.load(self.checkpoint_path + '/model_R_best.pt')   ## model_R_best 
            self.model.restoration_network.load_state_dict(R_state_dict['model_R_state_dict'])
            print(f'load restoration network status from {self.checkpoint_path}/model_R_11.pt, epoch: {R_state_dict["epoch"]}')   ## model_R_best
            save_epoch = R_state_dict["epoch"]
        return save_epoch

    def load_best_stage1_model(self):
        path = self.checkpoint_path.replace(f'model_stage{self.config.stage}', 'model_stage1')
        state_dict = torch.load(path + '/model_D_best.pt')
        self.model.degradation_learning_network.load_state_dict(state_dict['model_D_state_dict'])
        # print('train',self.optimizer_D.load_state_dict(state_dict[]))
        # print('train',state_dict)
        # self.optimizer_D.load_state_dict(state_dict['optimizer_D_state_dict'])
        # self.scheduler_D.load_state_dict(state_dict['scheduler_D_state_dict'])
        print(f'load degradation learning network status  from {path}/model_D_best.pt, epoch: {state_dict["epoch"]}')

    def train(self, dataloader, train_log, global_step):
        self.model.train()
        report = Train_Report()
        start = time.time()

        for idx, (lr_blur_seq, hr_sharp_seq, lr_sharp_seq, flow) in enumerate(dataloader):
            lr_blur_seq = lr_blur_seq.cuda()
            hr_sharp_seq = hr_sharp_seq.cuda()
            lr_sharp_seq = lr_sharp_seq.cuda()
            flow = flow.cuda()

            result_dict = self.model(lr_blur_seq, hr_sharp_seq)

            batch_size, _, t, _, _ = lr_blur_seq.shape

            # pretrain degradation learning network
            if self.config.stage == 1:
                recon_loss = self.criterion(result_dict['recon'], lr_blur_seq[:, :, t//2, :, :])
                hr_warping_loss = self.config.hr_warping_loss_weight * self.criterion(result_dict['hr_warp'], hr_sharp_seq[:, :, t//2:t//2+1, :, :].repeat([1,1,t,1,1]))
                # RAFT pseudo-GT optical flow loss
                flow_loss = self.config.flow_loss_weight * self.criterion(result_dict['image_flow'], flow)
                # TA loss for degradation learning network
                D_TA_loss = self.config.D_TA_loss_weight * self.criterion(result_dict['F_sharp_D'], lr_sharp_seq)

                total_loss = recon_loss + hr_warping_loss + flow_loss + D_TA_loss

                self.optimizer_D.zero_grad()
                total_loss.backward()
                self.optimizer_D.step()

                report.update(batch_size, 0, recon_loss.item(), hr_warping_loss.item(), 0, flow_loss.item(), D_TA_loss.item(), 0, total_loss.item())

            # train full network
            elif self.config.stage == 2:
                restoration_loss = self.criterion(result_dict['output'], hr_sharp_seq[:, :, t//2, :, :])
                recon_loss = self.config.Net_D_weight * self.criterion(result_dict['recon'], lr_blur_seq[:, :, t//2, :, :])
                lr_warping_loss = self.config.lr_warping_loss_weight * self.criterion(result_dict['lr_warp'], lr_blur_seq[:, :, t//2:t//2 + 1, :, :].repeat([1,1,t,1,1]))
                hr_warping_loss = self.config.Net_D_weight * self.config.hr_warping_loss_weight * self.criterion(result_dict['hr_warp'], hr_sharp_seq[:, :, t//2:t//2+1, :, :].repeat([1,1,t,1,1]))
                # RAFT pseudo-GT optical flow loss
                flow_loss = self.config.Net_D_weight * self.config.flow_loss_weight * self.criterion(result_dict['image_flow'], flow)
                # TA loss for degradation learning network and restoration network
                R_TA_loss = self.config.R_TA_loss_weight * self.criterion(result_dict['F_sharp_R'], lr_sharp_seq)
                D_TA_loss = self.config.Net_D_weight * self.config.D_TA_loss_weight * self.criterion(result_dict['F_sharp_D'], lr_sharp_seq)
                
                total_loss = restoration_loss + recon_loss + hr_warping_loss + lr_warping_loss + flow_loss + R_TA_loss + D_TA_loss

                self.optimizer_D.zero_grad()
                self.optimizer_R.zero_grad()
                total_loss.backward()
                self.optimizer_D.step()
                self.optimizer_R.step()

                report.update(batch_size, restoration_loss.item(), recon_loss.item(), hr_warping_loss.item(), lr_warping_loss.item(), flow_loss.item(), D_TA_loss.item(), R_TA_loss.item(), total_loss.item())

            global_step += 1

            if global_step % 500 == 0 or idx == len(dataloader) - 1:
                lr_D = self.scheduler_D.optimizer.state_dict()['param_groups'][0]['lr']
                lr_R = self.scheduler_R.optimizer.state_dict()['param_groups'][0]['lr'] if self.config.stage == 2 else None

                period_time = time.time() - start
                now_time = f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}]-'
                prefix_str = f'[{global_step}/{len(dataloader) * self.config.num_epochs}]  '
                result_str = report.result_str(lr_D, lr_R, period_time)

                train_log.write(now_time + prefix_str + result_str)
                start = time.time()
                report.__init__()

                if self.config.save_train_img:
                    if self.config.stage == 1:
                        src = [lr_blur_seq[:, :, t // 2, :, :], result_dict['recon']]
                    elif self.config.stage == 2:
                        src = [lr_blur_seq[:, :, t // 2, :, :], result_dict['recon'], result_dict['output'], hr_sharp_seq[:, :, t // 2, :, :]]
                    self.save_manager.save_batch_images(src, batch_size, global_step)

        self.scheduler_D.step()
        if self.config.stage == 2:
            self.scheduler_D.step()

        return global_step

    def validate(self, dataloader, val_log, epoch):
        self.model.eval()
        report = Train_Report()
        start = time.time()

        with torch.no_grad():
            for idx, (lr_blur_seq, hr_sharp_seq, lr_sharp_seq, flow) in enumerate(dataloader):
                lr_blur_seq = lr_blur_seq.cuda()
                hr_sharp_seq = hr_sharp_seq.cuda()
                lr_sharp_seq = lr_sharp_seq.cuda()
                flow = flow.cuda()

                result_dict = self.model(lr_blur_seq, hr_sharp_seq)

                batch_size, _, t, _, _ = lr_blur_seq.shape

                if self.config.stage == 1:
                    recon_loss = self.criterion(result_dict['recon'], lr_blur_seq[:, :, t // 2, :, :])
                    hr_warping_loss = self.config.hr_warping_loss_weight * self.criterion(result_dict['hr_warp'], hr_sharp_seq[:, :, t // 2:t // 2 + 1, :, :].repeat([1, 1, t, 1, 1]))
                    flow_loss = self.config.flow_loss_weight * self.criterion(result_dict['image_flow'], flow)
                    D_TA_loss = self.config.D_TA_loss_weight * self.criterion(result_dict['F_sharp_D'], lr_sharp_seq)

                    total_loss = recon_loss + hr_warping_loss + flow_loss + D_TA_loss
                    report.update(batch_size, 0, recon_loss.item(), hr_warping_loss.item(), 0, flow_loss.item(), D_TA_loss.item(), 0, total_loss.item())
                    report.update_recon_metric(result_dict['recon'], lr_blur_seq[:, :, t // 2, :, :])

                elif self.config.stage == 2:
                    restoration_loss = self.criterion(result_dict['output'], hr_sharp_seq[:, :, t // 2, :, :])
                    recon_loss = self.config.Net_D_weight * self.criterion(result_dict['recon'], lr_blur_seq[:, :, t // 2, :, :])
                    lr_warping_loss = self.config.lr_warping_loss_weight * self.criterion(result_dict['lr_warp'], lr_blur_seq[:, :, t // 2:t // 2 + 1, :, :].repeat([1, 1, t, 1, 1]))
                    hr_warping_loss = self.config.Net_D_weight * self.config.hr_warping_loss_weight * self.criterion(result_dict['hr_warp'], hr_sharp_seq[:, :, t // 2:t // 2 + 1, :, :].repeat([1, 1, t, 1, 1]))
                    flow_loss = self.config.Net_D_weight * self.config.flow_loss_weight * self.criterion(result_dict['image_flow'], flow)
                    R_TA_loss = self.config.R_TA_loss_weight * self.criterion(result_dict['F_sharp_R'], lr_sharp_seq)
                    D_TA_loss = self.config.Net_D_weight * self.config.D_TA_loss_weight * self.criterion(result_dict['F_sharp_D'], lr_sharp_seq)

                    total_loss = restoration_loss + recon_loss + hr_warping_loss + lr_warping_loss + flow_loss + R_TA_loss + D_TA_loss
                    report.update(batch_size, restoration_loss.item(), recon_loss.item(), hr_warping_loss.item(), lr_warping_loss.item(), flow_loss.item(), D_TA_loss.item(), R_TA_loss.item(), total_loss.item())
                    report.update_recon_metric(result_dict['recon'], lr_blur_seq[:, :, t // 2, :, :])
                    report.update_restoration_metric(result_dict['output'], hr_sharp_seq[:, :, t//2, :, :])

        period_time = time.time() - start
        prefix_str = f'[{epoch}/{self.config.num_epochs}]'
        result_str = report.val_result_str(period_time)

        val_log.write(prefix_str + result_str)

        if self.config.stage == 1:
            return report.recon_psnr
        elif self.config.stage == 2:
            return report.psnr

    def test(self, dataloader):
        from utils import denorm
        self.model.eval()
        self.save_manager = SaveManager(self.config)

        with torch.no_grad():
            for idx, (lr_blur_seq, filename) in enumerate(dataloader):
                lr_blur_seq = lr_blur_seq.cuda()


                modelS_ours = self.model.cuda()
                inputs = torch.rand((2, 3, 5, 256//4, 256//4)).cuda()
                modelS_ours = modelS_ours.cuda()
                inputs = inputs.unsqueeze(0)
                macs, param = (profile(modelS_ours, inputs))
                print('[FCVSR-S] FLOPs (G):', 2*macs/1000/1000/1000, 'Param (M):', param/1000/1000)
                print('lr_blur_seq',lr_blur_seq.shape) #   1 3 3  180 320 

                # print('lr_blur_seq',lr_blur_seq) 
                preprocess_time = time.time()
                result_dict = self.model(lr_blur_seq)
                print('time',time.time()-preprocess_time)
                # sys.exit(0)

                output = result_dict['output']

                output = output.squeeze(dim=0)
                output = denorm(output)
                # print('output',output)  

                filename = filename[0]
                filepath = os.path.basename(os.path.dirname(filename))
                filename = os.path.basename(filename)
                # filename = os.path.join(self.config.save_dir, 'test', filepath, filename)
                filename = os.path.join(self.config.save_dir, self.config.save_dataset, filepath, filename)
                self.save_manager.save_image(output, filename)
                print('Save Image:', idx,'...', end="\r")

    def test_quantitative_result(self, epoch, gt_dir, output_dir, image_border):
        import cv2
        import glob

        report = TestReport(epoch, output_dir)
        scene_list = sorted(glob.glob(os.path.join(gt_dir, '*')))
        # print('scene_list',scene_list)

        for scene in scene_list:
            scene_name = os.path.basename(scene)
            filelist = sorted(glob.glob(os.path.join(scene, '*.png')))
            report.scene_init(scene_name)
            for filename in filelist[image_border:-image_border]:
                gt_img = cv2.imread(filename)
                output_img = cv2.imread(os.path.join(output_dir, scene_name, os.path.basename(filename)))
                report.update_metric(gt_img, output_img, os.path.basename(filename))
                print('Image PSNR:', filename,'...', end="\r")
            report.scene_del(scene_name)






import numpy as np

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]   # 
    rad = np.sqrt(np.square(u) + np.square(v))
    # print("u v",u.shape, v.shape)
    a = np.arctan2(-v, -u)/np.pi
    # print("a",a.shape)
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        # ch_idx =  2-i
        flow_image[:,:,ch_idx] = np.floor(255 * col)  # 255
        # flow_image[:,:,ch_idx] = np.floor(col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)

def flowbatch_to_image(flowbatch_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    # assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    # assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    flowbatch_uv = np.transpose(flowbatch_uv, (1,2,3,0))
    # print('flowbatch_uv',flowbatch_uv.shape)
    T, H, W, C = flowbatch_uv.shape
    flowrgb = np.zeros([T, H, W, 3])
    for i in range(T):
        flow_uv = flowbatch_uv[i,:,:,:]
        if clip_flow is not None:
            flow_uv = np.clip(flow_uv, 0, clip_flow)
        u = flow_uv[:,:,0]
        v = flow_uv[:,:,1]
        rad = np.sqrt(np.square(u) + np.square(v))
        rad_max = np.max(rad)
        epsilon = 1e-5
        u = u / (rad_max + epsilon)
        v = v / (rad_max + epsilon)
        flow_rgb = flow_uv_to_colors(u, v, convert_to_bgr)
        # print('flow_rgb',flow_rgb.shape)
        flowrgb[i,:,:, :] = flow_rgb
    
    flowrgb = np.transpose(flowrgb, (3, 0, 1, 2))
    return flowrgb


