# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path

import hydra
from omegaconf import DictConfig, ListConfig

os.environ['DETECTRON2_DATASETS'] = str(Path.home() / 'data/detectron2')
### Detectron2 expects the following directory structure:
# coco/
# ├── annotations/
# │   ├── instances_{train,val}2017.json
# │   └── person_keypoints_{train,val}2017.json
# └── {train,val}2017/

import logging
import math
import random
import shutil

import tqdm

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from contextlib import ExitStack, contextmanager

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from detectron2.config import CfgNode, LazyConfig, get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data.detection_utils import read_image
## Test
from detectron2.evaluation import COCOEvaluator
from detectron2.projects.point_rend import add_pointrend_config
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import loratorch as lora
from compressai.zoo import image_models
from examples.utils.alignment import Alignment
## General
from examples.utils.dataloader import MSCOCO, Kodak
from examples.utils.predictor import ModPredictor
from examples.weight_entropy_module import *


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


## Function for model to eval
@contextmanager
def inference_context(model):
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
    
    def psnr(self, output, target):
        mse = torch.mean((output - target) ** 2)
        if(mse == 0):
            return 100
        max_pixel = 1.
        psnr = 10 * torch.log10(max_pixel / mse)
        return torch.mean(psnr)

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["rdloss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]        
        out["psnr"] = self.psnr(torch.clamp(output["x_hat"],0,1), target)
        return out


class TaskLoss(nn.Module):
    def __init__(self, cfg, det_cfg, device) -> None:
        super().__init__()
        self.task_net = ModPredictor(det_cfg).model.backbone
        for p in self.task_net.parameters():
            p.requires_grad = False
        self.task_net.eval()
        
        if isinstance(det_cfg, CfgNode):
            self.pixel_mean = torch.Tensor([103.530, 116.280, 123.675]).view(-1, 1, 1).to(device)
            self.pixel_std = torch.Tensor([1, 1, 1]).view(-1, 1, 1).to(device)
        else:
            self.pixel_mean = torch.Tensor([123.675, 116.280, 103.530]).view(-1, 1, 1).to(device)
            self.pixel_std = torch.Tensor([58.395, 57.120, 57.375]).view(-1, 1, 1).to(device)
        
        if 'rcnn' in cfg.task_model:
            self.scale = 0.2
        elif 'detr' == cfg.task_model:
            self.scale = 16.0
        elif 'point_rend' == cfg.task_model:
            self.scale = 0.02
        else:
            raise ValueError("Invalid task model type")
        
    def normalize(self, x):
        return (x - self.pixel_mean) / self.pixel_std

    def forward(self, output, d):
        with torch.no_grad():
            ## Ground truth for perceptual loss
            d = d.flip(1).mul(255)
            d = self.normalize(d)
            gt_out = self.task_net(d)
        
        x_hat = torch.clamp(output["x_hat"], 0, 1)
        x_hat = x_hat.flip(1).mul(255)
        x_hat = self.normalize(x_hat)
        task_net_out = self.task_net(x_hat)
        
        return self.scale * sum(nn.MSELoss(reduction='mean')(x1, x2) for x1, x2 in zip(gt_out.values(), task_net_out.values()))


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, cfg):
    """Set optimizer for only the parameters for propmts"""

    parameters = {
        name
        for name, param in net.named_parameters()
        if param.requires_grad and not name.endswith(".quantiles")
    }    
    if cfg.mode == 'TTT':
        lora.mark_only_lora_as_trainable_singular_only(net)      
    else:
        lora.mark_only_lora_as_trainable(net)
    
    params_dict = dict(net.named_parameters())    

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=cfg.learning_rate,
    )    

    return optimizer    


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    np.random.seed(seed)
    random.seed(seed)


def train_one_epoch(train_dataloader, optimizer, model, criterion_rd, criterion_task, lmbda, cfg):
    totalloss = AverageMeter() 
    model.train()    
    device = next(model.parameters()).device
    if cfg.mode == 'TTT':
        tqdm_emu = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=True, leave=True)
    else:
        tqdm_emu = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for i, d in tqdm_emu:
        d = d.to(device)
        optimizer.zero_grad()        
        out_net = model(d)

        out_criterion = criterion_rd(out_net, d)
        perc_loss = criterion_task(out_net, d)
        total_loss = perc_loss + lmbda * out_criterion['bpp_loss']
        total_loss.backward()
        optimizer.step()

        update_txt=f'[{i*len(d)}/{len(train_dataloader.dataset)}] | Loss: {total_loss.item():.3f} | Distortion loss: {perc_loss.item():.5f} | Bpp loss: {out_criterion["bpp_loss"].item():.4f} | lr: {optimizer.param_groups[0]["lr"]}'
        if cfg.mode in ['TTT', 'test']:
            print(update_txt)
            totalloss.update(total_loss.detach())
        tqdm_emu.set_postfix_str(update_txt, refresh=True)

    return totalloss.avg


def validation_epoch(epoch, val_dataloader, model, criterion_rd, criterion_task, lmbda):
    model.eval()
    device = next(model.parameters()).device

    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    psnr = AverageMeter()
    percloss = AverageMeter()
    totalloss = AverageMeter()

    with torch.no_grad():
        tqdm_meter = tqdm.tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        for i, d in tqdm_meter:
            align = Alignment(divisor=256, mode='resize').to(device)
            d = d.to(device)
            align_d = align.align(d)

            out_net = model(align_d)
            out_criterion = criterion_rd(out_net, align_d)
            perc_loss = criterion_task(out_net, align_d)
            total_loss = perc_loss + lmbda * out_criterion['bpp_loss']

            bpp_loss.update(out_criterion["bpp_loss"])
            mse_loss.update(out_criterion["mse_loss"])
            psnr.update(out_criterion['psnr'])
            percloss.update(perc_loss)
            totalloss.update(total_loss)

            txt = f"Loss: {totalloss.avg:.3f} | MSE loss: {mse_loss.avg:.5f} | Perception loss: {percloss.avg:.4f} | Bpp loss: {bpp_loss.avg:.4f}"
            tqdm_meter.set_postfix_str(txt)
        
    model.train()
    print(f"Epoch: {epoch} | bpp loss: {bpp_loss.avg:.5f} | psnr: {psnr.avg:.5f}")
    return totalloss.avg

def crop_image_grid(image_tensor, crop_size=256):
    _, _, H, W = image_tensor.shape

    crops = []
    for i in range(0, H, crop_size):
        for j in range(0, W, crop_size):
            top = i
            left = j
            bottom = min(i + crop_size, H)
            right = min(j + crop_size, W)            
            crop = image_tensor[0, :, top:bottom, left:right]
            if crop.shape[1] != crop_size or crop.shape[2] != crop_size:
                crop = F.pad(crop, (0, max(0, crop_size - crop.shape[2]), 0, max(0, crop_size - crop.shape[1]))) # type: ignore            
            crops.append(crop)
    return crops

def test_epoch_w_instance_train(test_dataloader, model_not_use, criterion_rd, predictor, evaluator, optimizer_not_use, taskcriterion, cfg, train_dataloader):
    spike_and_slap_cdf = SpikeAndSlabCDF()
    weight_entropyModule = WeightEntropyModule(spike_and_slap_cdf)
    weight_entropyModule.eval()

    device = "cuda" if cfg.cuda and torch.cuda.is_available() else "cpu"    
    weight_entropyModule.to(device)

    bpp_loss = AverageMeter()
    psnr = AverageMeter()        
    w_bpp_loss = AverageMeter()

    image_ids = []
    for i, batch in enumerate(test_dataloader):
        with ExitStack() as stack:
            image_ids.append(batch[0]['image_id'])
            
            align = Alignment(divisor=256, mode='resize').to(device)
            d = torch.stack([batch[0]['image'].to(device).float().div(255)]).flip(1)
            align_d = align.align(d)

            set_seed(cfg.seed)
            model = image_models[cfg.model](quality=int(cfg.quality_level), prompt_config=cfg)
            model = model.to(device) 

            if cfg.checkpoint_backbone: 
                ckpt_path = cfg.checkpoint_backbone.format(quality_level=cfg.quality_level)
                logging.info("Loading "+ckpt_path)
                checkpoint = torch.load(ckpt_path, map_location=device)
                
                if list(checkpoint["state_dict"].keys())[0][:7]=='module.':
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in checkpoint["state_dict"].items():
                        name = k[7:] 
                        new_state_dict[name] = v
                else:
                    new_state_dict = checkpoint['state_dict']
                model.load_state_dict(new_state_dict, strict=False)

            if cfg.checkpoint_pre_trained: 
                logging.info("Loading "+str(cfg.checkpoint_pre_trained))
                checkpoint = torch.load(cfg.checkpoint_pre_trained, map_location=device)
                
                if list(checkpoint["state_dict"].keys())[0][:7]=='module.':
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in checkpoint["state_dict"].items():
                        name = k[7:] 
                        new_state_dict[name] = v
                else:
                    new_state_dict = checkpoint['state_dict']
                model.load_state_dict(new_state_dict, strict=False)
                
            optimizer = configure_optimizers(model, cfg)
            if cfg.scheduler == 'multistep':
                lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25, 30, 35], gamma=0.5)   # for 40 epochs  
            elif cfg.scheduler == 'cosine':
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)
            elif cfg.scheduler == 'cosine_warmup':
                lr_scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=cfg.epochs//5, cycle_mult=1.0, max_lr=cfg.learning_rate, min_lr=cfg.learning_rate/100, warmup_steps=cfg.epochs//10, gamma=0.5)
            elif cfg.scheduler == 'constant':
                lr_scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=cfg.epochs)
            else:
                raise ValueError("Invalid scheduler type")

            if i == 0: 
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        print(f"Parameter: {name}, Shape: {param.shape}")
            
            if cfg.train_image_form == 0:
                temp_dataloader = DataLoader(align_d, batch_size=len(align_d), shuffle=False) # type: ignore
            elif cfg.train_image_form == 1:
                resized_align_d = crop_image_grid(align_d, crop_size=256)
                temp_dataloader = DataLoader(resized_align_d, batch_size=len(resized_align_d), shuffle=False) # type: ignore
            else:
                raise ValueError("Invalid train_image_form")

            tqrange = tqdm.trange(0, cfg.epochs, desc=f'Image {i+1}/{len(test_dataloader)}')
            best_loss = float("inf")
            print(f'{batch[0]["file_name"]}')
            for epoch in tqrange:
                if cfg.VPT_lmbda:
                    VPT_lmbda = cfg.VPT_lmbda
                else:
                    VPT_lmbda = cfg.VPT_lmbdas[cfg.quality_level - 1]
                loss = train_one_epoch(temp_dataloader, optimizer, model, criterion_rd, taskcriterion, VPT_lmbda, cfg)
                lr_scheduler.step()
                is_best = loss < best_loss
                best_loss = min(loss, best_loss)
                if is_best:
                    best_state_dict = model.state_dict()
            model.load_state_dict(best_state_dict, strict=False)  
            torch.cuda.empty_cache()

            new_state_dict_w = model.state_dict()
            w_bpp_sum = 0
            if cfg.mode == 'TTT':
                for key in model.state_dict():
                    if 'lora_S' in key and 'g_s' in key:  
                        N, _, H, W = d.size()
                        weight = model.state_dict()[key] 
                        og_shape = weight.shape
                        weight = weight.reshape(1,1,-1)
                        
                        w_hat, w_likelihood = weight_entropyModule(weight, False)
                        w_hat = w_hat.reshape(og_shape)

                        w_bpp = torch.log(w_likelihood) / (-math.log(2) * H * W)
                        w_bpp_sum += w_bpp.sum()
                        new_state_dict_w[key] = w_hat

            model.load_state_dict(new_state_dict_w, strict=False)  
            model.eval()
            if isinstance(predictor.model, nn.Module):
                stack.enter_context(inference_context(predictor.model))
            stack.enter_context(torch.no_grad())

            img = read_image(batch[0]["file_name"], format="BGR")

            out_net = model(align_d)
            out_net['x_hat'] = align.resume(out_net['x_hat']).clamp_(0, 1) # type: ignore
            out_criterion = criterion_rd(out_net, d)

            trand_y_tilde = out_net['x_hat'].flip(1).mul(255)

            bpp_loss.update(out_criterion["bpp_loss"])
            psnr.update(out_criterion['psnr'])
            w_bpp_loss.update(w_bpp_sum)

            predictions = predictor(trand_y_tilde, img.shape[0], img.shape[1])
            evaluator.process(batch, [predictions])

            del d, align_d, out_net, trand_y_tilde
            torch.cuda.empty_cache()                        

        txt = f"Bpp loss: {bpp_loss.avg:.4f} | PSNR loss: {psnr.avg:.4f}"
        # tqdm_meter.set_postfix_str(txt)

    results = evaluator.evaluate(image_ids)
    model.train()
    print(f"bpp loss(+ wbpp): {bpp_loss.avg + w_bpp_loss.avg:.4f} | bpp loss: {bpp_loss.avg:.5f} | psnr: {psnr.avg:.5f}")    
    return

def test_epoch(test_dataloader, model, criterion_rd, predictor, evaluator):
    model.eval()
    device = next(model.parameters()).device

    bpp_loss = AverageMeter()
    psnr = AverageMeter()

    image_ids = []
    with torch.no_grad():        
        tqdm_meter = tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        for i, batch in tqdm_meter:
            with ExitStack() as stack:
                image_ids.append(batch[0]['image_id'])

                if isinstance(predictor.model, nn.Module):
                    stack.enter_context(inference_context(predictor.model))
                stack.enter_context(torch.no_grad())

                align = Alignment(divisor=256, mode='resize').to(device)

                img = read_image(batch[0]["file_name"], format="BGR")
                d = torch.stack([batch[0]['image'].to(device).float().div(255)]).flip(1)
                align_d = align.align(d)

                out_net = model(align_d)
                out_net['x_hat'] = align.resume(out_net['x_hat']).clamp_(0, 1) # type: ignore
                out_criterion = criterion_rd(out_net, d)

                trand_y_tilde = out_net['x_hat'].flip(1).mul(255)

                bpp_loss.update(out_criterion["bpp_loss"])
                psnr.update(out_criterion['psnr'])

                ## MaskRCNN
                predictions = predictor(trand_y_tilde, img.shape[0], img.shape[1])
                evaluator.process(batch, [predictions])
            txt = f"Bpp loss: {bpp_loss.avg:.4f} | PSNR loss: {psnr.avg:.4f}"
            tqdm_meter.set_postfix_str(txt)

    results = evaluator.evaluate(image_ids)
    model.train()
    print(f"bpp loss: {bpp_loss.avg:.5f} | psnr: {psnr.avg:.5f}")
    return


def save_checkpoint(state, is_best, base_dir, filename="checkpoint.pth.tar"):
    ckpt_path = os.path.join(base_dir, filename)
    logging.info(f"Saving checkpoint: {ckpt_path}")
    torch.save(state, ckpt_path)
    if is_best:
        logging.info(f"Saving BEST checkpoint: {ckpt_path}")
        shutil.copyfile(ckpt_path, os.path.join(base_dir, "checkpoint_best_loss.pth.tar"))


@hydra.main(config_path='../config', config_name='detection')
def main(cfg: DictConfig):

    base_dir = cfg.root
    
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)
    device = "cuda" if cfg.cuda and torch.cuda.is_available() else "cpu"

    if cfg.dataset=='coco':
        if cfg.task_model == 'detr':
            det_cfg = LazyConfig.load(cfg.detr_config_path)
            det_cfg.weights = cfg.detr_ckpt_url # type: ignore
            tasks = ["bbox"]
        else:
            det_cfg = get_cfg() # get default cfg
            if cfg.task_model == 'faster_rcnn':
                cfg_path = os.path.join(cfg.config_path, 'faster_rcnn_R_50_FPN_3x.yaml')
                ckpt_path = os.path.join(cfg.config_path, 'model_final_280758.pkl')
                tasks = ["bbox"]
            elif cfg.task_model == 'mask_rcnn':
                cfg_path = os.path.join(cfg.config_path, 'mask_rcnn_R_50_FPN_3x.yaml')
                ckpt_path = os.path.join(cfg.config_path, 'model_final_f10217.pkl')
                tasks = ["segm"]
            elif cfg.task_model == 'point_rend':
                add_pointrend_config(det_cfg)
                cfg_path = cfg.point_rend_config_path
                ckpt_path = cfg.point_rend_ckpt_url
                tasks = ["segm"]
            else:
                raise ValueError("Invalid task model type")
            det_cfg.merge_from_file(cfg_path)
            det_cfg.MODEL.WEIGHTS = ckpt_path
    
        det_transformer = transforms.Compose([
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        ## Training
        if cfg.hyperparameter_tuning:  
            #for hyper-parameter tuning
            train_dataset = MSCOCO(cfg.dataset_path+"/train2017/",
                                det_transformer,
                                cfg.example_path + "/utils/img_list_hyper_10000.txt")        
        else :        
            train_dataset = MSCOCO(cfg.dataset_path+"/train2017/",
                                det_transformer,
                                cfg.example_path + "/utils/img_list.txt")
                                #    "./utils/img_list.txt")
      
        val_dataset = MSCOCO(cfg.dataset_path+"/val2017/",
                            transforms.ToTensor(),
                            cfg.example_path + "/utils/image_list_part_of_val2017.txt")        

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=cfg.batch_size,
                                      num_workers=cfg.num_workers,
                                      shuffle=True,
                                      pin_memory=(device=="cuda"))
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=cfg.test_batch_size,
                                    num_workers=cfg.num_workers,
                                    shuffle=False,
                                    pin_memory=(device=="cuda"))
        
        ## Testing
        if cfg.mode in ['TTT', 'test']:
            evaluator = COCOEvaluator('coco_2017_val', tasks, False)
            evaluator.reset()
            
            test_dataloader = build_detection_test_loader(get_cfg(), 'coco_2017_val') # type: ignore
            if isinstance(cfg.num_images, ListConfig):
                subset_indices = list(range(cfg.num_images[0], cfg.num_images[1]))
            else:
                subset_indices = list(range(cfg.num_images))
            subset = Subset(test_dataloader.dataset, subset_indices) # type: ignore
            test_dataloader = DataLoader(subset, batch_size=1, shuffle=False, collate_fn=test_dataloader.collate_fn) # type: ignore        
            predictor = ModPredictor(det_cfg)
        
    net = image_models[cfg.model](quality=int(cfg.quality_level), prompt_config=cfg)
    net = net.to(device)

    optimizer = configure_optimizers(net, cfg)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1)
    rdcriterion = RateDistortionLoss(lmbda=cfg.lmbda)
    taskcriterion = TaskLoss(cfg, det_cfg, device)

    last_epoch = 0
    if cfg.checkpoint_backbone:
        ckpt_path = cfg.checkpoint_backbone.format(quality_level=cfg.quality_level) 
        logging.info("Loading "+ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        if list(checkpoint["state_dict"].keys())[0][:7]=='module.':
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                name = k[7:] 
                new_state_dict[name] = v
        else:
            new_state_dict = checkpoint['state_dict']
        net.load_state_dict(new_state_dict, strict=False)

    if cfg.checkpoint_pre_trained: 
        logging.info("Loading "+str(cfg.checkpoint_pre_trained))
        checkpoint = torch.load(cfg.checkpoint_pre_trained, map_location=device)
        
        if list(checkpoint["state_dict"].keys())[0][:7]=='module.':
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                name = k[7:] 
                new_state_dict[name] = v
        else:
            new_state_dict = checkpoint['state_dict']
        net.load_state_dict(new_state_dict, strict=False)

    if cfg.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    if cfg.mode in ['TTT', 'test']:
        if cfg.mode == 'TTT':
            test_epoch_w_instance_train(test_dataloader, net, rdcriterion, predictor, evaluator, optimizer, taskcriterion, cfg, train_dataloader)
        else:
            test_epoch(test_dataloader, net, rdcriterion, predictor, evaluator)
        return
    
    if cfg.VPT_lmbda:
        VPT_lmbda = cfg.VPT_lmbda
    else:
        VPT_lmbda = cfg.VPT_lmbdas[cfg.quality_level - 1]

    best_loss = validation_epoch(-1, val_dataloader, net, rdcriterion, taskcriterion, VPT_lmbda)
    tqrange = tqdm.trange(last_epoch, cfg.epochs)
    for epoch in tqrange:
        train_one_epoch(train_dataloader, optimizer, net, rdcriterion, taskcriterion, VPT_lmbda, cfg)
        loss = validation_epoch(epoch, val_dataloader, net, rdcriterion, taskcriterion, VPT_lmbda)
        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if cfg.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                base_dir,
                filename='checkpoint.pth.tar'
            )
            if epoch%10==9:
                ckpt_path = os.path.join(base_dir, f"checkpoint.pth.tar")
                ckpt_epoch_path = os.path.join(base_dir, f"checkpoint_{epoch}.pth.tar")
                shutil.copyfile(ckpt_path, ckpt_epoch_path)
    

if __name__ == "__main__":
    main()
