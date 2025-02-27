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

import hydra
from omegaconf import ListConfig, OmegaConf
import wandb

import tqdm
import math
import random
import shutil
import sys
import os
import logging

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.optim as optim

import torchvision
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import transforms
from torchvision.models import resnet50

from compressai.zoo import image_models

import loratorch as lora
from weight_entropy_module import *


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
    
    def psnr(self, output, target):
        mse = torch.mean((output - target) ** 2, dim=(1, 2, 3))
        # if (mse == 0):
        #     return 100
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

class FeatureHook():
    def __init__(self, module):
        module.register_forward_hook(self.attach)
    
    def attach(self, model, input, output):
        self.feature = output

class Clsloss(nn.Module):
    def __init__(self, device, perceptual_loss=False) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.classifier = resnet50(True)
        self.classifier.requires_grad_(False)
        self.hooks = [FeatureHook(i) for i in [
            self.classifier.layer1,
            self.classifier.layer2,
            self.classifier.layer3,
            self.classifier.layer4,
        ]]
        self.classifier = self.classifier.to(device)
        for k, p in self.classifier.named_parameters():
            p.requires_grad = False
        self.classifier.eval()
        self.perceptual_loss = perceptual_loss
        self.transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def forward(self, output, d, y_true=None):
        if y_true is None:
            assert self.perceptual_loss
        x_hat = torch.clamp(output["x_hat"],0,1)
        pred = self.classifier(self.transform(x_hat))
        loss = self.ce(pred, y_true)
        accu = sum(torch.argmax(pred,-1)==y_true)/pred.shape[0]
        if self.perceptual_loss:
            pred_feat = [i.feature.clone() for i in self.hooks]
            _ = self.classifier(self.transform(d))
            ori_feat = [i.feature.clone() for i in self.hooks]
            perc_loss = torch.stack([nn.functional.mse_loss(p,o, reduction='none').mean((1,2,3)) for p,o in zip(pred_feat, ori_feat)])
            perc_loss = perc_loss.mean()
            return loss, accu, perc_loss

        return loss, accu, None
    
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


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    np.random.seed(seed)
    random.seed(seed)


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


def train_one_epoch(
    model, criterion_rd, criterion_cls, train_dataloader, optimizer, lmbda, cfg,
):
    totalloss = AverageMeter()
    model.train()
    device = next(model.parameters()).device
    if cfg.mode == 'TTT':
        tqdm_emu = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=True, leave=True)
    else:
        tqdm_emu = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for i, (d,l) in tqdm_emu:
        d = d.to(device)
        l = l.to(device)

        optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion_rd(out_net, d)
        loss, accu, perc_loss = criterion_cls(out_net, d, l)
        total_loss = 1000*lmbda*perc_loss + out_criterion['bpp_loss']
        total_loss.backward()
        optimizer.step()

        update_txt=f'[{i*len(d)}/{len(train_dataloader.dataset)}] | Loss: {total_loss.item():.3f} | Distortion loss: {perc_loss.item():.5f} | Bpp loss: {out_criterion["bpp_loss"].item():.4f} | lr: {optimizer.param_groups[0]["lr"]}'
        if cfg.mode in ['TTT', 'test']:
            print(update_txt)
            totalloss.update(total_loss.detach())
        tqdm_emu.set_postfix_str(update_txt, refresh=True)
        
    # wandb.log({'train/total_loss': totalloss.avg})
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


def test_epoch(epoch, test_dataloader, model, criterion_rd, criterion_cls, lmbda, stage='test'):
    model.eval()
    device = next(model.parameters()).device

    loss_am = AverageMeter()
    percloss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()
    accuracy = AverageMeter()
    totalloss = AverageMeter()

    with torch.no_grad():
        tqdm_meter = tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        for i, (d,l) in tqdm_meter:
            d = d.to(device)
            l = l.to(device)
            out_net = model(d)
            out_criterion = criterion_rd(out_net, d)
            loss, accu, perc_loss = criterion_cls(out_net, d, l)

            bpp_loss.update(out_criterion["bpp_loss"])
            psnr.update(out_criterion['psnr'])
            accuracy.update(accu)

    model.train()
    print(f"{epoch} | bpp loss: {bpp_loss.avg:.5f} | psnr: {psnr.avg:.5f} | accu: {accuracy.avg:.5f}")
    return loss_am.avg


def test_epoch_w_instance_train(test_dataloader, criterion_rd, criterion_cls, cfg):
    spike_and_slap_cdf = SpikeAndSlabCDF()
    weight_entropyModule = WeightEntropyModule(spike_and_slap_cdf)
    weight_entropyModule.eval()
    
    device = 'cuda' if cfg.cuda and torch.cuda.is_available() else 'cpu'
    weight_entropyModule.to(device)

    bpp_loss = AverageMeter()
    accuracy = AverageMeter()
    w_bpp_loss = AverageMeter()

    for i, (d, l) in enumerate(test_dataloader):
        set_seed(cfg.seed)
        
        model = image_models[cfg.model](quality=int(cfg.quality_level), prompt_config=cfg)
        model = model.to(device)
        
        if cfg.checkpoint_pre_trained:
            ckpt_path = cfg.checkpoint_pre_trained.format(quality_level=cfg.quality_level)
            logging.info("Loading "+str(ckpt_path))
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
            
        if cfg.checkpoint: 
            logging.info("Loading "+str(cfg.checkpoint))
            checkpoint = torch.load(cfg.checkpoint, map_location=device)
            
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
        elif cfg.scheduler == 'constant':
            lr_scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=cfg.epochs)
        else:
            raise ValueError("Invalid scheduler type")
        
        if i == 0: 
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"Parameter: {name}, Shape: {param.shape}")
                    
        tqrange = tqdm.trange(0, cfg.epochs, desc=f'Image {i+1}/{len(test_dataloader)}')
        best_loss = float("inf")
        temp_dataloader = DataLoader(TensorDataset(d, l), batch_size=len(d), shuffle=False)
        for epoch in tqrange:
            if cfg.VPT_lmbda:
                VPT_lmbda = cfg.VPT_lmbda
            else:
                VPT_lmbda = cfg.VPT_lmbdas[cfg.quality_level - 1]
            loss = train_one_epoch(model, criterion_rd, criterion_cls, temp_dataloader, optimizer, VPT_lmbda, cfg)
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
        
        with torch.no_grad():
            for d, l in temp_dataloader:
                d = d.to(device)
                l = l.to(device)
                out_net = model(d)
                out_criterion = criterion_rd(out_net, d)
                _, accu, _ = criterion_cls(out_net, d, l)
                accuracy.update(accu)
        
        bpp_loss.update(out_criterion["bpp_loss"])
        w_bpp_loss.update(w_bpp_sum)
        accuracy.update(accu)
        torch.cuda.empty_cache()
        
    print(f"bpp loss(+ wbpp): {bpp_loss.avg + w_bpp_loss.avg:.6f} | bpp loss: {bpp_loss.avg:.6f} | accu: {accuracy.avg:.6f}")
    return      
                

def save_checkpoint(state, is_best, base_dir, filename="checkpoint.pth.tar"):
    ckpt_path = os.path.join(base_dir, filename)
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copyfile(ckpt_path, os.path.join(base_dir, "checkpoint_best_loss.pth.tar"))


@hydra.main(config_path='config', config_name='classification_svdlora')
def main(cfg):
    # wandb.init(
    #     project="SVD LoRA",
    #     config=OmegaConf.to_container(cfg, resolve=True), # type: ignore
    # ) 
    
    base_dir = cfg.root        
    
    msg = f'======================= {cfg.root} ======================='
    logging.info(msg)
    for k in cfg.__dict__:
        logging.info(k + ':' + str(cfg.__dict__[k]))
    logging.info('=' * len(msg))

    cls_transforms = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()]
    )

    if cfg.dataset=='imagenet':
        train_dataset = torchvision.datasets.ImageNet(cfg.dataset_path,split='train', transform=cls_transforms)
        test_dataset = torchvision.datasets.ImageNet(cfg.dataset_path,split='val', transform=cls_transforms)
        
        val_dataset,_ = torch.utils.data.random_split(test_dataset,[15000,35000])
        small_train_datasets = torch.utils.data.random_split(train_dataset,[40000]*32+[1167])

    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)
    device = "cuda" if cfg.cuda and torch.cuda.is_available() else "cpu"

    val_dataloader = DataLoader(val_dataset,batch_size=cfg.test_batch_size,num_workers=cfg.num_workers,shuffle=False,pin_memory=(device == "cuda"),)
    if isinstance(cfg.num_images, ListConfig):
        subset_indices = list(range(cfg.num_images[0], cfg.num_images[1]))
    else:
        subset_indices = list(range(cfg.num_images))
    test_dataset = Subset(test_dataset, subset_indices)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.test_batch_size,num_workers=cfg.num_workers, shuffle=False, pin_memory=(device == "cuda"))

    net = image_models[cfg.model](quality=int(cfg.quality_level), prompt_config=cfg)
    net = net.to(device)

    optimizer = configure_optimizers(net, cfg)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60], gamma=0.5)
    rdcriterion = RateDistortionLoss(lmbda=cfg.lmbda)
    clscriterion = Clsloss(device, perceptual_loss=True)

    last_epoch = 0
    if cfg.checkpoint_pre_trained:
        ckpt_path = cfg.checkpoint_pre_trained.format(quality_level=cfg.quality_level) 
        logging.info("Loading "+str(ckpt_path))
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
    
    if cfg.checkpoint: 
        logging.info("Loading "+str(cfg.checkpoint))
        checkpoint = torch.load(cfg.checkpoint, map_location=device)
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
        
    if cfg.VPT_lmbda:
        VPT_lmbda = cfg.VPT_lmbda
    else:
        VPT_lmbda = cfg.VPT_lmbdas[cfg.quality_level - 1]
    
    if cfg.mode in ['TTT', 'test']:
        if cfg.mode == 'TTT':
            test_epoch_w_instance_train(test_dataloader, rdcriterion, clscriterion, cfg)
        else:
            test_epoch(-1, test_dataloader, net, rdcriterion,clscriterion, VPT_lmbda,'test')
        return

    best_loss = float("inf")
    tqrange = tqdm.trange(last_epoch, cfg.epochs)
    loss = test_epoch(-1, val_dataloader, net, rdcriterion,clscriterion, VPT_lmbda,'val')
    for epoch in tqrange:
        train_dataloader = DataLoader(
        small_train_datasets[epoch%32],
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
        )
        train_one_epoch(
            net,
            rdcriterion,
            clscriterion,
            train_dataloader,
            optimizer,
            VPT_lmbda,
            cfg
        )
        loss = test_epoch(epoch, val_dataloader, net, rdcriterion, clscriterion, VPT_lmbda,'val')
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
                ckpt_path = os.path.join(base_dir, "checkpoint.pth.tar")
                ckpt_epoch_path = os.path.join(base_dir, f"checkpoint_{epoch}.pth.tar")
                shutil.copyfile(ckpt_path, ckpt_epoch_path)
    


if __name__ == "__main__":
    main()
