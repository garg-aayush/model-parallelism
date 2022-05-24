# Model parallelism 
Training large models, especially for 3D image segmentation/reconstruction problems, can lead to out-of-memory when the size of the model is too large for a single GPU. To train such large models, layers can be pipelined across different GPU devices as described in [torchgpipe](https://torchgpipe.readthedocs.io/en/stable/gpipe.html). However, pipelining models, such as [ResNets](https://arxiv.org/abs/1512.03385) & [UNets](https://arxiv.org/abs/1505.04597), can be difficult due to the skip connections between different layers.

This repository provides two examples on how one can do model parallelism for architectures (ResNets, UNets) with skip conections using [torchgpipe](https://torchgpipe.readthedocs.io/en/stable/gpipe.html) [skip](https://github.com/kakaobrain/torchgpipe/tree/master/torchgpipe/skip) module:
- [A_resnet18_model_sharding.py](https://github.com/garg-aayush/model-parallelism/blob/main/A_resnet18_model_sharding.py) : It uses the MNIST example to show short skip connections implementation for ResNets
- [B_unet_model_sharding.py](https://github.com/garg-aayush/model-parallelism/blob/main/A_unet_model_sharding.py) : It uses the Kaggle's CARAVANA image masking challenge example to show long skip/cat connections implementation for UNets

## Quickstart
#### Setup the environment
```
# clone project
git clone https://https://github.com/garg-aayush/model-parallelism
cd model-parallelism

# create conda environment
conda create -n sharding python=3.6
conda activate sharding

# install requirements
pip install -r requirements.txt
```

#### Running the code
```
## Assumes access to 2 GPUs

# run resnet model sharding example
python A_resnet18_model_sharding.py

# run unet model sharding example
# assumes CARAVANA dataset (use download_data.sh) is downloaded to correct datapath
python B_unet_model_sharding.py
```

#### Downloading the CARAVANA dataset
```
## run
bash download_data.sh
```
or download it manually from [CARAVANA](https://www.kaggle.com/competitions/carvana-image-masking-challenge/data)


#### Folder structure
```
  model_parallelism/
  │
  ├── A_resnet18_model_sharding.py : resnet18 model sharding example
  ├── B_unet_model_sharding : unet model sharding example
  │
  ├── ResNet.py : resnet18 model implemented using torchgpipe for model parallelism
  ├── UNet.py : unet model implemented using torchgpipe for model parallelism
  │
  ├── download_data.sh : script to download CARAVANA dataset from kaggle
  │
  ├── data/ - placeholder folder for input data
  │
  ├── FIG/ - validation unet results images at each epoch for qc 
  │
  └── requirements.txt : file to install python dependencies
 ```

## Torchgpipe
[Torchgpipe](https://torchgpipe.readthedocs.io/en/stable/gpipe.html) implements model parallelism by spliting the model into multiple partitions and placing each partition on a different device (GPU) to occupy more memory capacity and pipeline parallelism by splitting a mini-batch into multiple micro-batches to make the devices work as parallel as possible. Note, [torchgpipe](https://torchgpipe.readthedocs.io/en/stable/gpipe.html) requires the model to be sequential, therefore, always wrap your model in `nn.Sequential` module.

### Skip-connections
Since torchgpipe requires the models to be sequential for partitioning. It makes use of `@skippable` class decorator to stash (store) and pop the tensors for use in later layers. Example:
```python
## Stash the tensor from Identity layer
@skippable(stash=['identity'])
class Identity(nn.Module):
    def forward(self, tensor: Tensor) -> Tensor:  # type: ignore
        yield stash('identity', tensor)
        return tensor

## Pop and use the tensor in Residual layer
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
 ```

#### Model balancing and micro-batches
Torchgpipe requires the user to provide model balance for each device, i.e. number of layers on individual devices. It is hard task to find the optimal balance of a model such that each device use similar memory load. However, as a starting point, use `torchgpipe.balance` for automatic balancing. It will give a good balance to start. After that, one can play around to find the optimal balance that gives best memory partition and least runtime per epoch. Example:
```python
from torchgpipe import GPipe
from torchgpipe.balance import balance_by_time

partitions = torch.cuda.device_count()
sample = torch.rand(128, 1, 28, 28)
balance = balance_by_time(partitions, model, sample)
model = GPipe(model, balance, chunks=8)
```
Using a smaller micro-batchwa help to reduce the bubble time (idle time) as partition wait for data from prior partition. However, a very small micro-batch can affect the model performance and GPU efficiency. Always, play around with number of micro-batches (defined by `chunks` parameter in `torchgpipe.Gpipe`) to come up with a final value.

There are many more features that are available in torchgpipe. See, https://torchgpipe.readthedocs.io/en/stable/gpipe.html for more elaborate information.

## Fairscale implementation
[Fairscale](https://fairscale.readthedocs.io/en/latest/deep_dive/pipeline_parallelism.html) also has a Gpipe implementation which has been adopted from torchgpipe. One can use the Fairscale implementation just by importing the same classes from `fairscale.nn.pipe`.

*Note, the fairscale implementation branch will be in added later.* 

## Feedback
To give feedback or ask a question or for environment setup issues, you can use the [Github Discussions](https://https://github.com/garg-aayush/pytorch-pl-hydra-templates/discussions).
