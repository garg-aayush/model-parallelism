"""ResNet-18 Model sharding example"""
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
from torch.utils.data import DataLoader
import torchvision
from torchgpipe.balance import balance_by_time

from ResNet import resnet18
from torchgpipe import GPipe
from utils import set_random_seed


################################################
# Global variables 
################################################
EXPERIMENT = 'resnet_mnist_sharding_2gpu'

BATCH_SIZE = 256
NUM_CLASSES = 10
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
RANDOM_SEED=42

CHUNKS = 4  # No of micro-batchesm the batch will be divided into 
AUTO_BALANCE = True # Automatically determine the number of layers on each GPU 
BALANCE = [5, 5] # Number of layers on each GPU (assuming 10 nn.Sequential layers and 2 GPUs)


# Dataloader
def dataloaders(batch_size: int=128, num_workers:int = 4) -> Tuple[DataLoader, DataLoader]:
    train_dataset = torchvision.datasets.MNIST(root='data', 
                               train=True, 
                               transform=torchvision.transforms.ToTensor(),
                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='data', 
                              train=False, 
                              transform=torchvision.transforms.ToTensor())


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

# train
def train(model: nn.Module, dataloader: DataLoader, epoch: int, in_device: str, out_device: str) -> float:
    torch.cuda.synchronize(in_device)

    steps = len(dataloader)
    loss_sum = torch.zeros(1, device=out_device)
    model.train()
    for i, (input, target) in enumerate(dataloader):
        input = input.to(device=in_device, non_blocking=True)
        target = target.to(device=out_device, non_blocking=True)

        output = model(input)
        loss = F.cross_entropy(output, target)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        loss_sum += loss.detach()

        if not i % 50:
            print (f'Epoch (train): {(epoch+1):3d}/{NUM_EPOCHS:3d} | batch {i:4d}/{steps:4d} | loss: {loss:1.4f}') 


    torch.cuda.synchronize(in_device)

    train_loss = loss_sum.item() / steps
    torch.cuda.synchronize(in_device)

    return train_loss


# eval
def evaluate(model: nn.Module, dataloader: DataLoader, in_device: str, out_device: str) -> Tuple[float, float]:
        tick = time.time()
        steps = len(dataloader)
        loss_sum = torch.zeros(1, device=out_device)
        accuracy_sum = torch.zeros(1, device=out_device)
        num_examples = 0
        model.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(dataloader):
                input = input.to(device=in_device)
                target = target.to(device=out_device)
                output = model(input)
                loss = F.cross_entropy(output, target)
                loss_sum += loss.detach()

                _, predicted = torch.max(output, 1)
                correct = (predicted == target).sum()
                accuracy_sum += correct
                num_examples += target.size(0)

                if not i % 50:
                    print (f'Epoch (val): {(epoch+1):3d}/{NUM_EPOCHS:3d} | batch {i:4d}/{steps:4d} | loss: {loss:1.4f}') 

        loss = loss_sum / steps
        accuracy = accuracy_sum / num_examples

        torch.cuda.synchronize(in_device)
        return loss.item(), accuracy.item()


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
    model = resnet18(num_classes=NUM_CLASSES, in_dim=1)
    model = cast(nn.Sequential, model)
    
    if AUTO_BALANCE:
        print('AUTO BALANCE')
        partitions = torch.cuda.device_count()
        print('No of GPUS: ', partitions)
        # MNIST: (1,28,28)
        sample = torch.rand(BATCH_SIZE,1,28,28)
        BALANCE = balance_by_time(partitions, model, sample)
    else:
        print('MANUAL BALANCE')
    print('Balance: ', BALANCE)
    
    model = GPipe(model, balance=BALANCE, chunks=CHUNKS)

    # In and Out devices
    in_device = model.devices[0]
    out_device = model.devices[-1]
    print('in_device: ', in_device)
    print('out_device: ', out_device)

    # Prepare dataloaders.
    train_dataloader, valid_dataloader = dataloaders(batch_size=BATCH_SIZE)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  

    # Train and evaluate
    total_time = time.time()
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        # train
        train_loss = train(model, train_dataloader, epoch, in_device, out_device)
        # evaluate
        valid_loss, valid_accuracy = evaluate(model, valid_dataloader, in_device, out_device)
        #print info
        epoch_time = ((time.time() - start_time)/60)
        print(f'Epoch: {epoch:3d}/{NUM_EPOCHS:3d} | train_loss: {train_loss:1.4f} | valid_loss: {valid_loss:1.4f} | valid_acc: {valid_accuracy:1.4f} | time: {epoch_time:2.4f} mins')
        
    total_time = ((time.time() - total_time)/60)
    print(f'Total Training Time: {total_time:2.4f} mins')
