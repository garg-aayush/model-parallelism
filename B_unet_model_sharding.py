"""UNet Model sharding example"""
import os
################################################
# Setup the visible GPUS
################################################
# For example, here I set GPUs 0 & 1 visible
# To run this example, make sure you have atleast 2 GPUS
# `nvidia-smi`: shows you available GPUs along with other info
#################################################
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

import platform
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, cast
from typing import Iterator

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
import torchvision
from torchgpipe.balance import balance_by_time
from torch import optim

from UNet import unet
from torchgpipe import GPipe


import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import set_random_seed, CarvanaDataset, plot_img_and_mask_val, dice_loss
from utils import multiclass_dice_coeff, dice_coeff
from tqdm import tqdm

################################################
# Global variables 
################################################
EXPERIMENT = 'resnet_mnist_sharding_2gpu'

BATCH_SIZE = 16
CHUNKS = 4 # No of micro-batchesm the batch will be divided into 
AUTO_BALANCE = False # Number of layers on each GPU (assuming 22 nn.Sequential layers and 4 GPUs)
BALANCE = [13,9] # Automatically determine the number of layers on each GPU
# # for 4 GPUs
# BALANCE = [5,9,4,4]
# BATCH_SIZE = 32
# CHUNKS = 4

NUM_CLASSES = 2
NUM_EPOCHS = 20
LEARNING_RATE = 1e-5
WEIGHT_DECAY=1e-8
MOMENTUM=0.9

RANDOM_SEED=42
VAL_PERCENT = 0.2
IMG_SCALE = 0.5
DIR_IMG = './data/CARAVANA/data/imgs'
DIR_MASK = './data/CARAVANA/data/masks'
FIG_DIR = './FIG/'

# dataloader
def dataloaders(batch_size: int=128, num_workers:int = 4, 
                img_scale: float=0.5, val_percent: float=0.1,
                random_seed: int=42,
                dir_img:str=None, dir_mask:str=None) -> Tuple[DataLoader, DataLoader]:
    
    dataset = CarvanaDataset(dir_img, dir_mask, img_scale)

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_dataset, test_dataset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(random_seed))

    train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True,
                          pin_memory=True,
                          drop_last=True,
                          num_workers=num_workers )

    test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size,
                         shuffle=False,
                         pin_memory=True,
                         drop_last=False,
                         num_workers=num_workers )
    
    return train_loader, test_loader

#train
def train(model: nn.Module, dataloader: DataLoader, epoch: int, in_device: str, out_device: str) -> float:
    torch.cuda.synchronize(in_device)

    steps = len(dataloader)
    loss_sum = torch.zeros(1, device=out_device)
    model.train()
    with tqdm(iterable=dataloader, total=steps, desc=f'Epoch {epoch}/{NUM_EPOCHS}', unit='batch') as pbar:
        for batch in pbar:
        #for i, batch in enumerate(dataloader):
            input = batch['image'].to(device=in_device, non_blocking=True)
            target = batch['mask'].to(device=out_device, non_blocking=True)

            output = model(input)
            loss = criterion(output, target) + dice_loss(F.softmax(output, dim=1).float(),
                                                F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float(), multiclass=True)


            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            loss_sum += loss.detach()
            pbar.set_postfix(**{'loss (batch)': loss.item()})
        
    torch.cuda.synchronize(in_device)

    train_loss = loss_sum.item() / steps
    torch.cuda.synchronize(in_device)

    return train_loss

#eval
def evaluate(model: nn.Module, dataloader: DataLoader, in_device: str, out_device: str, epoch:int) -> Tuple[float, float]:
    torch.cuda.synchronize(in_device)
    num_val_batches = len(dataloader)
    loss_sum = 0
    dice_score = 0
    batch_idx = 0 
    model.eval()
    # iterate over the validation set
    with tqdm(dataloader, total=num_val_batches, desc='Val', unit='batch', leave=False) as pbar:
        for batch in pbar:
            input = batch['image'].to(device=in_device, non_blocking=True, dtype=torch.float32)
            target = batch['mask'].to(device=out_device, non_blocking=True, dtype=torch.long)

            with torch.no_grad():
                # predict the mask
                output = model(input)
                loss = criterion(output, target) + dice_loss(F.softmax(output, dim=1).float(),
                                                F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float(), multiclass=True)

                target = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()

                # convert to one-hot format
                if NUM_CLASSES == 1:
                    output = (F.sigmoid(output) > 0.5).float()
                    # compute the Dice score
                    dice_score += dice_coeff(output, mask_true, reduce_batch_first=False)
                else:
                    output = F.one_hot(output.argmax(dim=1), NUM_CLASSES).permute(0, 3, 1, 2).float()
                    # compute the Dice score, ignoring background
                    dice_score += multiclass_dice_coeff(output[:, 1:, ...], target[:, 1:, ...], reduce_batch_first=False)
            
                loss_sum += loss.detach()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
            
                if batch_idx == 0:
                    plot_img_and_mask_val(input[0:4].permute(0,2,3,1).cpu().numpy(), target[0:4].cpu().numpy(), output[0:4].cpu().numpy(), FIG_DIR, epoch)
            
                batch_idx+=1

            
    val_loss = loss_sum.item() / num_val_batches 
    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches, val_loss


if __name__ == '__main__':
    
    # SEED
    set_random_seed(RANDOM_SEED)

    # HEADER
    print(f'{EXPERIMENT} \n\nchunks: {CHUNKS:2d}  \nbatch_size: {BATCH_SIZE:4d} \nepochs: {NUM_EPOCHS:3d}\n')
    print(f'python: {platform.python_version()}')
    print(f'torch: {torch.__version__}')
    print(f'cudnn: {torch.backends.cudnn.version()}')
    print(f'cuda: {torch.version.cuda}\n')

    # Model
    model = unet()
    model = cast(nn.Sequential, model)
    if AUTO_BALANCE:
        print('AUTO BALANCE')
        partitions = torch.cuda.device_count()
        print('No of GPUS: ', partitions)
        sample = torch.rand(BATCH_SIZE,3,572,572)
        BALANCE = balance_by_time(partitions, model, sample)
    else:
        print('MANUAL BALANCE')

    print('Balance: ', BALANCE)
    #model = GPipe(model, balance, devices=DEVICES, chunks=CHUNKS)
    model = GPipe(model, balance=BALANCE, chunks=CHUNKS)

    # In and Out devices
    in_device = model.devices[0]
    out_device = model.devices[-1]
    #torch.cuda.set_device(in_device)
    print('Balance: ', BALANCE)
    print('in_device: ', in_device)
    print('out_device: ', out_device)

    # Prepare dataloaders.
    train_dataloader, valid_dataloader = dataloaders(batch_size=BATCH_SIZE,
                                                    num_workers=4,
                                                    img_scale=IMG_SCALE,
                                                    val_percent=VAL_PERCENT,
                                                    random_seed=RANDOM_SEED,
                                                    dir_img=DIR_IMG,
                                                    dir_mask=DIR_MASK)

    # Optimizer
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    criterion = nn.CrossEntropyLoss()
    global_step = 0


    total_time = time.time()
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        # train
        train_loss = train(model, train_dataloader, epoch, in_device, out_device)
        # # evaluate
        dice_score, valid_loss = evaluate(model, valid_dataloader, in_device, out_device, epoch)
        #print info
        epoch_time = ((time.time() - start_time)/60)
        print(f'Epoch: {epoch:3d}/{NUM_EPOCHS:3d} | train_loss: {train_loss:1.4f} | valid_loss: {valid_loss:1.4f} | dice_score: {dice_score:1.4f} | time: {epoch_time:2.4f} mins')
        
    total_time = ((time.time() - total_time)/60)
    print(f'Total Training Time: {total_time:2.4f} mins')

