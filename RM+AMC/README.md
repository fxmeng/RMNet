# AMC Pruning
This example shows us how to use AMCPruner example.

## Step 1: train a model for pruning
Run following command to train a mobilenetv2 model:
```bash
python3 amc_train.py --model_type mobilenetv2 --dataset cifar10
```
Once finished, saved checkpoint file can be found at:
```
logs/mobilenetv2_cifar10_train-run1/ckpt.best.pth
```

## Step 2: Pruning with AMCPruner
Run following command to prune the trained model:
```bash
python3 amc_search.py --model_type mobilenetv2 --dataset cifar10 --ckpt logs/mobilenetv2_cifar10_train-run1/ckpt.best.pth --flops_ratio 0.5
```
Once finished, pruned model and mask can be found at:
```
logs/mobilenetv2_cifar10_r0.5_search-run2
```

## Step 3: Finetune pruned model
Run `amc_train.py` again with `--ckpt` and `--mask` to speedup and finetune the pruned model:
```bash
python3 amc_train.py --model_type mobilenetv2 --dataset cifar10 --ckpt logs/mobilenetv2_cifar10_r0.5_search-run2/best_model.pth --mask logs/mobilenetv2_cifar10_r0.5_search-run2/best_mask.pth
```
Once finished, saved checkpoint file can be found at:
```
logs/mobilenetv2_cifar10_finetune-run4/ckpt.best.pth
```

# RM + AMC pruning

## Step 1: train a model for pruning
Run following command to train a mobilenetv2 model:
```bash
python3 amc_train.py --model_type mobilenetv2
```
Once finished, saved checkpoint file can be found at:
```
logs/mobilenetv2_cifar10_train-run1/ckpt.best.pth
```

## Step 2: Converting mobilenetv2 to mobilenetv1 and finetune this model
Run `amc_train.py` again with `--ckpt` and `--mask` to speedup and finetune the pruned model:
```bash
python3 amc_train.py --model_type mobilenetv1 --dataset cifar10 --ckpt logs/mobilenetv2_cifar10_train-run1/ckpt.best.pth
```
Once finished, saved checkpoint file can be found at:
```
logs/mobilenetv1_cifar10_finetune-run2/ckpt.best.pth
```

## Step 3: Pruning with AMCPruner
Run following command to prune the trained model:
```bash
python3 amc_search.py --model_type mobilenetv1 --dataset cifar10 --ckpt logs/mobilenetv1_cifar10_finetune-run2/ckpt.best.pth --flops_ratio 0.5
```
Once finished, pruned model and mask can be found at:
```
logs/mobilenetv1_cifar10_r0.5_search-run3
```

## Step 4: Finetune pruned model
Run `amc_train.py` again with `--ckpt` and `--mask` to speedup and finetune the pruned model:
```bash
python3 amc_train.py --model_type mobilenetv1 --dataset cifar10 --ckpt logs/mobilenetv1_cifar10_r0.5_search-run3/best_model.pth --mask logs/mobilenetv1_cifar10_r0.5_search-run3/best_mask.pth
```
Once finished, saved checkpoint file can be found at:
```
logs/mobilenetv1_cifar10_finetune-run4/ckpt.best.pth
```
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