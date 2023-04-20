# Resource Efficient Neural Networks through Hessian Based Pruning

HAP is an advanced structured pruning library written by the authors of [1] for PyTorch. HAP prunes channels based on their second-order sensitivity. Channels are sorted based on this metric, and only insensitive channels are pruned.

In this repository, we propose a modified approach for HAP to calculate the relative Hessian trace using FP16 instead of FP32. This has been shown to be faster and more GPU memory efficient than the traditional FP32 approach.


## Installation using Anaconda

- **To install HAP** and develop locally:

   ```
   git clone https://github.com/JackkChong/Resource-Efficient-Neural-Networks-Using-Hessian-Based-Pruning
   ```

- Create and import the conda environment
   ```
   conda env create --file environment.yml
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



3. Quantizing on CIFAR10

   ```
   python main_quantize.py --learning_rate 0.0001 --weight_decay 0.0001 --dataset cifar10 --epoch 30 --from_HAP True
   ```
   
## References

- [1] Shixing Yu, Zhewei Yao, Amir Gholami, Zhen Dong, Michael W. Ma-honey, and Kurt Keutzer. [Hessian-Aware Pruning and Optimal Neural Implant](https://arxiv.org/abs/2101.08940) CoRR, abs/2101.08940, 2021.
- [2] Zhewei Yao, Zhen Dong, Zhangcheng Zheng, Amir Gholami, Jiali Yu, Eric Tan, Leyuan Wang, Qijing Huang, Yida Wang, Michael W. Mahoney, and Kurt Keutzer. [HAWQ-V3: Dyadic Neural Network Quantization](https://arxiv.org/abs/2011.10680) CoRR, abs/2011.10680, 2020.


## License

This framework is released under the [MIT license](https://github.com/JackkChong/Resource-Efficient-Neural-Networks-Using-Hessian-Based-Pruning/blob/master/LICENSE.txt).
