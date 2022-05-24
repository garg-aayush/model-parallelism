# Model parallelism 
Training large models, especially for 3D image segmentation/reconstruction problems, can lead to out-of-memory when the size of the model is too large for a single GPU. To train such a large model, layers can be pipelined across different GPU devices as described in [torchgpipe](https://torchgpipe.readthedocs.io/en/stable/gpipe.html). However, pipelining a model, such as [ResNets](https://arxiv.org/abs/1512.03385) & [UNets](https://arxiv.org/abs/1505.04597), can be difficult due to the skip connections between different layers.

This repository provides examples on how can do model parallelism for architectures (ResNets, UNets) with skip conections using [torchgpipe](https://torchgpipe.readthedocs.io/en/stable/gpipe.html) [skip](https://github.com/kakaobrain/torchgpipe/tree/master/torchgpipe/skip) module. 

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
## Assumes you have access to 2 GPUs

# run resnet model sharding example
python A_resnet18_model_sharding.py

# run unet model sharding example
# assumes you have downloaded CARAVANA dataset (use download_data.sh) and have setup the correct datapath 
python B_unet_model_sharding.py
```

#### Folder structure
```
  model_parallelism/
  │
  ├── A_resnet18_model_sharding.py : resnet18 model sharding example
  ├── B_unet_model_sharding : unet model sharding example
  │
  ├── ResNet.py : resnet18 model implemented using pipe (torchgpipe) for model parallelism
  ├── UNet.py : unet model implemented using pipe (torchgpipe) for model parallelism
  │
  ├── download_data.sh : script to download CARAVANA dataset from kaggle
  │
  ├── data/ - placeholder folder for input data
  │
  ├── FIG/ - validation unet results images at each epoch for qc 
  │
  └── requirements.txt : file to install python dependencies
 ```
