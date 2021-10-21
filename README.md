# RMNet: Equivalently Removing Residual Connection from Networks

This repository is the official implementation of "RMNet: Equivalently Removing Residual Connection from Networks". 

## Requirements

To install requirements:

```setup
pip install torch
pip install torchvision
```

## Training

To train the models in the paper, run this command:

```train
python train.py -a rmnet41x5_16 --dist-url 'tcp://127.0.0.1:23333' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --workers 32 [imagenet-folder with train and val folders]
```


## Our Pre-trained Models

You can download pretrained models here:

- [Our pre-trained models](https://drive.google.com/drive/folders/1Mu3fXmZPm2EB9Bv17e41H3EfBOLlJYcw?usp=sharing) trained on ImageNet. 

## Evaluation

To evaluate our pre-trained models trained on ImageNet, run:

```eval
python train.py -a rmnet41x5_16 -e rmnet41x5_16.tar [imagenet-folder with train and val folders]
```

## Results

Our model achieves the following performance on :

### Image Classification on ImageNet
| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| RMNet 41x5\_16 |     78.498%   |      94.086% |
| RMNet 50x5\_32 |     79.076%   |      94.444% |
| RMNet 50x6\_32 |     79.57%    | 94.644% |
| RMNet 101x6\_16 |     80.07%    |      94.918% |
| RMNet 152x6\_32 |     80.356%   |      80.356% |



## Contributing

Our code is based on [RepVGG](https://github.com/DingXiaoH/RepVGG)