##########################
### MODEL
##########################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

from collections import OrderedDict
from typing import TYPE_CHECKING, Optional, Tuple, Union, Iterator

from torch import Tensor, nn

from torchgpipe.skip import Namespace, pop, skippable, stash

if TYPE_CHECKING:
    NamedModules = OrderedDict[str, nn.Module]
else:
    NamedModules = OrderedDict
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


@skippable(stash=['identity'])
class Identity(nn.Module):
    def forward(self, tensor: Tensor) -> Tensor:  # type: ignore
        yield stash('identity', tensor)
        return tensor


@skippable(pop=['identity'])
class Residual(nn.Module):
    """A residual block for ResNet."""

    def __init__(self, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.downsample = downsample

    def forward(self, input: Tensor) -> Tensor:  # type: ignore
        identity = yield pop('identity')
        if self.downsample is not None:
            identity = self.downsample(identity)
        return input + identity


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


def basicblock(inplanes: int,
               planes: int,
               stride: int = 1,
               downsample: Optional[nn.Module] = None,
               inplace: bool = False,
               ) -> nn.Sequential:
    """Creates a basicblock for ResNet18 as a :class:`nn.Sequential`."""

    layers: NamedModules = OrderedDict()

    ns = Namespace()
    layers['identity'] = Identity().isolate(ns)  # type: ignore

    layers['conv1'] = conv3x3(inplanes, planes, stride)
    layers['bn1'] = nn.BatchNorm2d(planes)
    layers['relu1'] = nn.ReLU(inplace=inplace)

    layers['conv2'] = conv3x3(planes, planes, stride=1)
    layers['bn2'] = nn.BatchNorm2d(planes)
    layers['residual'] = Residual(downsample).isolate(ns)  # type: ignore
    layers['relu2'] = nn.ReLU(inplace=inplace)

    return nn.Sequential(layers)


class InputLayer(nn.Module):
  def __init__(self, in_dim, out_dim, kernel_size=[7,3], stride=[2,2], padding=[3,1]):
    super().__init__()
    self.out = nn.Sequential(
                        nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size[0], stride=stride[0], padding=padding[0], bias=False),
                        nn.BatchNorm2d(out_dim),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=kernel_size[1], stride=stride[1], padding=padding[1])
    )
        
  def forward(self, x):
    return self.out(x)


class Downsample(nn.Module):
  def __init__(self, in_dim, out_dim, kernel_size=1, stride=2):
    super().__init__()
    self.out = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_dim),
                )
    
  def forward(self, x):
    return self.out(x)


class OutputLayer(nn.Module):
  def __init__(self, in_dim, out_dim, kernel_size=7, stride=2):
    super().__init__()
    #self.avgpool = nn.AvgPool2d(kernel_size, stride=stride)
    self.out = nn.Sequential(nn.Flatten(),
                        nn.Linear(in_dim, out_dim))
        
  def forward(self, x):
    # MNIST: nchannel=1
    #x = self.avgpool(x)
    return self.out(x)


def resnet18(num_classes : int = 10, in_dim : int = 1):
    model = nn.Sequential(OrderedDict([
                ("input_layer", InputLayer(in_dim, 64, kernel_size=[7,3], stride=[2,2], padding=[3,1])),
                ("basic_block11", basicblock(inplanes=64,planes=64,stride=1,downsample=None)),
                ("basic_block12", basicblock(inplanes=64,planes=64,stride=1,downsample=None)),
                ("basic_block21", basicblock(inplanes=64,planes=128,stride=2,downsample=Downsample(64,128,1,2))),
                ("basic_block22", basicblock(inplanes=128,planes=128,stride=1,downsample=None)),
                ("basic_block31", basicblock(inplanes=128,planes=256,stride=2,downsample=Downsample(128,256,1,2))),
                ("basic_block32", basicblock(inplanes=256,planes=256,stride=1,downsample=None)),
                ("basic_block41", basicblock(inplanes=256,planes=512,stride=2,downsample=Downsample(256,512,1,2))),
                ("basic_block42", basicblock(inplanes=512,planes=512,stride=1,downsample=None)),
                ("output_layer", OutputLayer(512, num_classes, kernel_size=7, stride=2)),
    ]))
    return model


