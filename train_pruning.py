import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import models 
import thop

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--sr', default=0, type=float, help='learning rate')
parser.add_argument('--threshold', default=0, type=float, help='learning rate')
parser.add_argument('--finetune', type=str)
parser.add_argument('--debn', action='store_true',default=False)
parser.add_argument('--eval', type=str)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='/dev/shm', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='/dev/shm', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
net = models.rmnet_pruning_18(10).to(device)
if args.sr*args.threshold==0:
    net.fix_mask()
if args.finetune or args.eval:
    if args.finetune:
        ckpt=torch.load(args.finetune)
    else:
        ckpt=torch.load(args.eval)
    net.load_state_dict(ckpt)
    net=net.cpu().prune(not args.debn).cuda()
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if args.sr*args.threshold>0 and not args.finetune:
           net.update_mask(args.sr,args.threshold)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print('Epoch: %d Acc: %.3f%%' %(epoch, 100.*correct/total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        if args.finetune:
            save_dir= args.finetune.replace('ckpt','finetune_lr%f'%args.lr)
        else:
            save_dir='./lr_%f_sr_%f_thres_%f'%( args.lr, args.sr,args.threshold)
            if not os.path.isdir(save_dir):
               os.mkdir(save_dir)
            save_dir+='/ckpt.pth'
        torch.save(net.state_dict(), save_dir)
        best_acc = acc

if args.eval:
    best_acc=100
    test(0)
    flops,params=thop.profile(net,(torch.randn(1,3,224,224).to(device),))
    print('flops:%.2fM,\tparams:%.2fM'%(flops/1e6,params/1e6))
else:
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)
        scheduler.step()
