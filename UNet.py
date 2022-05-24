##########################
### MODEL
##########################
# Simple GPU implementation and dice_loss function taken from https://github.com/milesial/Pytorch-UNet
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

from collections import OrderedDict
from typing import TYPE_CHECKING, Optional, Tuple, Union, Iterator, cast

from torch import Tensor, nn

from torchgpipe.skip import Namespace, pop, skippable, stash
from torchgpipe import GPipe
from torchgpipe.balance import balance_by_time

if TYPE_CHECKING:
    NamedModules = OrderedDict[str, nn.Module]
else:
    NamedModules = OrderedDict
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpTranspose(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.out = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        else:
            self.out = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2))

    def forward(self, x):
        return self.out(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



@skippable(stash=['cat'])
class UpStash(nn.Module):
    def forward(self, tensor: Tensor) -> Tensor:  # type: ignore
        yield stash('cat', tensor)
        return tensor


@skippable(pop=['cat'])
class UpCat(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input: Tensor) -> Tensor:  # type: ignore
        identity = yield pop('cat')
        diffY = identity.size()[2] - input.size()[2]
        diffX = identity.size()[3] - input.size()[3]

        input = F.pad(input, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        output = torch.cat([identity, input], dim=1)
        return output


def unet(n_channels=3, n_classes=2, bilinear=False):

    cat1 = Namespace()
    cat2 = Namespace()
    cat3 = Namespace()
    cat4 = Namespace()
    
    factor = 2 if bilinear else 1
    
    model = nn.Sequential(OrderedDict([
                ("InConv", DoubleConv(n_channels, 64)),
                ("cat1", UpStash().isolate(cat1)),
                ("down1", Down(64,128)),
                ("cat2", UpStash().isolate(cat2)),
                ("down2", Down(128, 256)),
                ("cat3", UpStash().isolate(cat3)),
                ("down3", Down(256, 512)),
                ("cat4", UpStash().isolate(cat4)),
                ("down4", Down(512, 1024 // factor)),
                ("UpT1", UpTranspose(1024, bilinear)),
                ("UpPop1", UpCat().isolate(cat4)),
                ("UpConv1", DoubleConv(1024, 512 // factor, 512 // (2*factor) if bilinear else None)),
                ("UpT2", UpTranspose(512, bilinear)),
                ("UpPop2", UpCat().isolate(cat3)),
                ("UpConv2", DoubleConv(512, 256 // factor, 256 // (2*factor) if bilinear else None)),
                ("UpT3", UpTranspose(256, bilinear)),
                ("UpPop3", UpCat().isolate(cat2)),
                ("UpConv3", DoubleConv(256,128 // factor, 128 // (2*factor) if bilinear else None)),
                ("UpT4", UpTranspose(128, bilinear)),
                ("UpPop4", UpCat().isolate(cat1)),
                ("UpConv4", DoubleConv(128,64, 128 // 2 if bilinear else None)),
                ("OutConv", OutConv(64, n_classes)),
    ]))
    return model

def flatten_sequential(module: nn.Sequential) -> nn.Sequential:
    """flatten_sequentials a nested sequential module."""
    if not isinstance(module, nn.Sequential):
        raise TypeError('not sequential')

    return nn.Sequential(OrderedDict(_flatten_sequential(module)))


def _flatten_sequential(module: nn.Sequential) -> Iterator[Tuple[str, nn.Module]]:
    for name, child in module.named_children():
        # flatten_sequential child sequential layers only.
        if isinstance(child, nn.Sequential):
            for sub_name, sub_child in _flatten_sequential(child):
                yield (f'{name}_{sub_name}', sub_child)
        else:
            yield (name, child)

# Helper functions
def set_random_seed(random_seed):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # model = unet()
    # x = torch.rand((128,3,572,572))
    # y = model(x)
    # print(y.shape)
    model = unet()
    model = cast(nn.Sequential, model)
    print('AUTO BALANCE')
    partitions = torch.cuda.device_count()
    print(partitions)
    BATCH_SIZE=16
    sample = torch.rand(BATCH_SIZE,3,640,959)
    BALANCE = balance_by_time(partitions, model, sample)
    print(BALANCE)
    model = GPipe(model, balance=BALANCE, chunks=4)
    print(model(sample.to(model.devices[0])).shape)
#print(model)
