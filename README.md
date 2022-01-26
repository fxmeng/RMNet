# RMNet: Equivalently Removing Residual Connection from Networks

This repository is the official implementation of "[RMNet: Equivalently Removing Residual Connection from Networks](https://arxiv.org/abs/2111.00687)". 

## Updates

Jan 25,2022, RM+AMC purning:

https://github.com/fxmeng/RMNet/blob/aec110b528c2646a19a20777bd5b93500e9b74a3/RM+AMC/README.md


Dec 24, 2021, RMNet Pruning:

`python train_pruning.py --sr xxx --threshold xxx`

`python train_pruning.py --eval xxx/ckpt.pth`

`python train_pruning.py --finetune xxx/ckpt.pth`

Nov 15, 2021, RM Opeartion now supports PreActResNet.

Nov 13, 2021, RM Opeartion now supports SEBlock.


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

## Evaluation

To evaluate our pre-trained models trained on ImageNet, run:

```eval
python train.py -a rmrep_69 -e checkpoint/rmrep_69.pth.tar [imagenet-folder with train and val folders]
```

## Results

Our model achieves the following performance on :

### Help pruning achieve better performance [Baidu Cloud(提取码:1jw2)](https://pan.baidu.com/s/1tCq7JWRKr3BuwgBlyF7ZPg )
| Method | Speed(Imgs/Sec) | Acc(%)|
| ----------------- | ----------------- | ---------- |
|Baseline|3752|71.79|
|AMC(0.75)|4873|70.94|
|AMC(0.7)|4949|70.84|
|AMC(0.5)|5483|68.89|
|RM+AMC(0.75)|5120|**73.21**|
|RM+AMC(0.7)|5238|72.63|
|RM+AMC(0.6)|5675|71.88|
|RM+AMC(0.5)|**6250**|71.01|

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


### Image Classification on ImageNet [Baidu Cloud(提取码:0mto)](https://pan.baidu.com/s/1FB7wyU52i_-EK4DnwRxfbQ). 
| Model name         | Top 1 Accuracy(%)  | Top 5 Accuracy(%) |
| ------------------ |---------------- | -------------- |
| RMNeXt 41x5\_16  |     78.498   |      94.086 |
| RMNeXt 50x5\_32  |     79.076   |      94.444 |
| RMNeXt 50x6\_32  |     79.57    |      94.644 |
| RMNeXt 101x6\_16 |     80.07    |      94.918 |
| RMNeXt 152x6\_32 |     80.356   |      80.356 |

## Citation

If you find this code useful, please cite the following paper:

```
@misc{meng2021rmnet,
      title={RMNet: Equivalently Removing Residual Connection from Networks}, 
      author={Fanxu Meng and Hao Cheng and Jiaxin Zhuang and Ke Li and Xing Sun},
      year={2021},
      eprint={2111.00687},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Contributing

Our code is based on [RepVGG](https://github.com/DingXiaoH/RepVGG) and [nni/amc pruning](https://github.com/microsoft/nni/tree/master/examples/model_compress/pruning/amc)
