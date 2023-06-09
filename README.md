
Pytorch implementation for **Forgery-Domain-Supervised Deepfake Detection with Non-negative Constraint**. [[Paper Link](https://ieeexplore.ieee.org/document/9839430)]

## Setup
Please install Pytorch first. This repo has been developed with python version 3.6, pytorch version 1.8.0, CUDA 10.2 and torchvision 0.9.0. For convenience, use [requirements](requirements.txt) to create a conda environment.

## Data preparation

We use the well-known public dataset `FaceForensics++` for training and evaluation.  Check the [github repo](https://github.com/ondyari/FaceForensics) and [paper link](https://arxiv.org/abs/1901.08971) for more details.


### Preprocessing

Following the settings, we use `dlib` to detect the face, enlarge the face region 1.3 times and crop it. The whole dataset contains 1000 videos, in which 720 for training, 140 for validation and 140 for testing. [Here](https://github.com/ondyari/FaceForensics/tree/master/dataset/splits) are split rules.

After preprocessing, the dataset should be organized as following:

```
|-- dataset
|   |-- train
|   |   |-- real
|   |   |	|-- 000
|   |   |	|	|-- frame0.jpg
|   |   |	|	|-- frame1.jpg
|   |   |	|	|-- ...
|   |   |	|-- 001
|   |   |	|-- ...
|   |   |-- fake
|   |   	|-- Deepfakes
|   |   	|	|-- 000_167
|   |		|	|	|-- frame0.jpg
|   |		|	|	|-- frame1.jpg
|   |		|	|	|-- ...
|   |		|	|-- 001_892
|   |		|	|-- ...
|   |   	|-- Face2Face
|   |		|	|-- ...
|   |   	|-- FaceSwap
|   |   	|-- NeuralTextures
|   |-- valid
|   |	|-- real
|   |	|	|-- ...
|   |	|-- fake
|   |		|-- ...
|   |-- test
|   |	|-- ...
```

### Usage
We recommend to use our implemented `get_FF_dataset` and  `get_FF_5class` functions to load pytorch datasets. Here's an simple example.

```python
train_dataset = get_FF_5_class(path_to_dataset, mode='train', isTrain=True)
valid_dataset = get_FF_dataset(path_to_dataset, mode='valid', isTrain=False, img_size=img_size, drop_rate=0.8)
```
The function `get_FF_5class` returns a 5-class label for multi-class supervision in our methods and `get_FF_dataset` returns vanilla binary label.

For more details about the function, param or customizing dataset, please refer to [dataset.md](dataset/dataset.md).

## Training

### Documentation
For a glimpse at the documentation training of our method, please run:
```python
python main.py --help
```
### Example
With default parameters setting, to traing a model with `AIM` and `NCC` from the scratch, please run:
```python
python main.py --output-dir /path/to/checkpoint/ --backbone-name resnet --dataset-path /path/to/dataset/ --use-ncc --use-aim --use-mc
```
If a pretrained backbone is loaded, we recommend to set `warmup-iters` to 20000.

## Evaluation

To evalulate on `FaceForensics++` dataset, please run:

```python
python evaluation.py --ckpt-path /path/to/checkpoint/ --backbone-name resnet --dataset-path /path/to/dataset/ --use-ncc --use-aim --use-mc
```

## Performance Analysis

We compare the baseline (vanilla xception, binary supervision) with our method (xception, multi-class supervision, with NCC and AIM).

The performance evaluation is conducted on `FF++ c40` test split. Execution time is measured with batch size 1, 20 runs averaged.

| Methods  | AUC |Params(M)|MACs(G)|CPU total(s)|GPU total(ms)|
|----------|-----|---------|-------|------------|-------------|
|Xception  |88.65|20.811   |6.012  |2.083       |121.743      |
|Ours      |92.15|37.605   |18.085 |2.404       |272.773      |

Note: Params and MACs are measured by [pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter). CPU total and GPU total are measured by [pytorch profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html). The CPU total time includes CUDALauchKernel. Check [comp_cost.py](comp_cost.py) and [runtime_cost.py](runtime_cost.py) for more details.
