<!-- <p align="center">
  <img src="imgs/neural_implant.png" width="800">
  <br />
  <br />
</p> -->

# Resource Efficient Neural Networks through Hessian Based Pruning

HAP is an advanced structured pruning library written for PyTorch. HAP prunes channels based on their second-order sensitivity. Channels are sorted based on this metric, and only insensitive channels are pruned.

## CIFAR10
- prune ratio = 0.770 for ResNet-32 to reach ~10% sparsity. <br />
- prune ratio = 0.80870 (FP32) for ResNet-56 to reach ~10% sparsity. <br />
- prune ratio = 0.760 for WideResNet-28-8 to reach 10% sparsity. <br />

## CIFAR100
- prune ratio = 0.780 for ResNet-32 to reach ~10% sparsity. <br />
- prune ratio = 0.81239 (FP16) and 0.825 (FP32) for ResNet-56 to reach ~10% sparsity. <br />
- prune ratio = 0.780 for WideResNet-28-8 to reach ~10% sparsity. <br />

## Installation using Anaconda

- Create a new environment for this project.
   ```
   conda create --name HAP_env
   ```
- Activate the new environment.
   ```
   conda activate HAP_env
   ```
- Install Pytorch (https://pytorch.org/get-started/locally/#start-locally)
   ```
   conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
   ```
- Install the following additional dependencies
   ```
   conda install -c conda-forge tqdm
   conda install -c conda-forge pytorch-model-summary
   conda install -c conda-forge progress
   ```
- **To install HAP** and develop locally:

```
git clone https://github.com/ICML2021Submission1958/Hessian-Aware-Pruning-and-Optimal-Neural-Implant
```

## Quick Start

1. Pretraining ResNet-32 on CIFAR10:

   ```
   python main_train.py --learning_rate 0.05 --weight_decay 0.001 --dataset cifar10 --epoch 300
   ```

   

2. Pruning on CIFAR10

   ```
   python main_prune.py --learning_rate 0.0512 --weight_decay 0.0005 --dataset cifar10 --epoch 300 --prune_ratio 0.77
   ```

   



## Related Work

- [Hessian-Aware Pruning and Optimal Neural Implant](https://arxiv.org/abs/2101.08940)





## License

HAP is released under the [MIT license](https://github.com/ICML2021Submission1958/Hessian-Aware-Pruning-and-Optimal-Neural-Implant/blob/main/LICENSE).
