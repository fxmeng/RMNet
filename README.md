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
python train.py -a rmrep_69 --dist-url 'tcp://127.0.0.1:23333' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --workers 32 [imagenet-folder with train and val folders]
```

## Our Pre-trained Models

You can download pretrained models here:

- [Our pre-trained models](https://drive.google.com/drive/folders/1Mu3fXmZPm2EB9Bv17e41H3EfBOLlJYcw?usp=sharing) trained on ImageNet. 

## Evaluation

To evaluate our pre-trained models trained on ImageNet, run:

```eval
python train.py -a rmrep_69 -e checkpoint/rmrep_69.pth.tar [imagenet-folder with train and val folders]
```

## Results

Our model achieves the following performance on :

### Help RepVGG achieve better performance even when the depth is large
| Arch                    | Top-1 Accuracy(%) | Top-5 Accuracy(%) | Train FLOPs(G) | Test FLOPs(M) |
| ----------------------- | ----------------- | ----------------- | ----------- | ---------- |
| RepVGG-21               | 72.508            | 90.840            | 2.4         | 2.1        |
| **RepVGG-21(RM 0.25)**  | **72.590**        | **90.924**        | **2.1**     | **2.1**    |
| RepVGG-37               | 74.408            | 91.900            | 4.4         | 4.0        |
| **RepVGG-37(RM 0.25)**  | **74.478**        | **91.892**        | **3.9**     | **4.0**    |
| RepVGG-69               | 74.526            | 92.182            | 8.6         | 7.7        |
| **RepVGG-69(RM 0.5)**   | **75.088**        | **92.144**        | **6.5**     | **7.7**    |
| RepVGG-133              | 70.912            | 89.788            | 16.8        | 15.1       |
| **RepVGG-133(RM 0.75)** | **74.560**        | **92.000**        | **10.6**    | **15.1**   |


### Image Classification on ImageNet
| Model name         | Top 1 Accuracy(%)  | Top 5 Accuracy(%) |
| ------------------ |---------------- | -------------- |
| RMNeXt 41x5\_16  |     78.498   |      94.086 |
| RMNeXt 50x5\_32  |     79.076   |      94.444 |
| RMNeXt 50x6\_32  |     79.57    |      94.644 |
| RMNeXt 101x6\_16 |     80.07    |      94.918 |
| RMNeXt 152x6\_32 |     80.356   |      80.356 |



## Contributing

Our code is based on [RepVGG](https://github.com/DingXiaoH/RepVGG)
