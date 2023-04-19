import os
import time
import argparse
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
import torchvision

from torch.utils.data import Subset
import random

from tensorboardX import SummaryWriter

from nets.pascal_resnet import *

from utils.preprocessing import *

from collections import defaultdict
from voc_eval import voc_eval

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from utils.preprocessing import voc_transform
import torch.multiprocessing as mp


# Training settings
parser = argparse.ArgumentParser(description='DoReFa-Net pytorch implementation')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='./VOCdevkit')
parser.add_argument('--log_name', type=str, default='pascal_resnet_w1a32')
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--pretrain_dir', type=str, default='./ckpt/resnet20_baseline')


parser.add_argument('--Wbits', type=int, default=4)
parser.add_argument('--Abits', type=int, default=32)

parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--wd', type=float, default=1e-4)

parser.add_argument('--train_batch_size', type=int, default=20)
parser.add_argument('--eval_batch_size', type=int, default=10)
parser.add_argument('--max_epochs', type=int, default=10)

parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--use_gpu', type=str, default='0')
parser.add_argument('--num_workers', type=int, default=5)

parser.add_argument('--cluster', action='store_true', default=False)

cfg = parser.parse_args()

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.use_gpu
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

voc_classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

GRAD_ACCUM_STEPS =2




def main():
  

    def train(epoch, model, criterion, optimizer, train_loader, summary_writer):
        print(f'\nEpoch: {epoch}')
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        for batch_idx, values in enumerate(train_loader):
            print("haha", batch_idx, len(train_loader))
            inputs, targets = values[:2]
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            if batch_idx % cfg.log_interval == 0:
                step = len(train_loader) * epoch + batch_idx
                duration = time.time() - start_time
                print('%s epoch: %d step: %d cls_loss= %.5f (%d samples/sec)' %
                    (datetime.now(), epoch, batch_idx, loss.item(),
                    cfg.train_batch_size * cfg.log_interval / duration))
                start_time = time.time()
                summary_writer.add_scalar('cls_loss', loss.item(), step)
                summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], step)

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == targets).float().sum(dim=1).eq(targets.size(1)).sum().item()
            total += targets.size(0)

        train_acc = 100. * correct / total
        summary_writer.add_scalar('train/loss', train_loss / (batch_idx + 1), epoch)
        summary_writer.add_scalar('train/accuracy', train_acc, epoch)
        print(f'Train Loss: {train_loss / (batch_idx + 1)} | Train Acc: {train_acc}%')


    def test(epoch, model, criterion, test_loader, summary_writer):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, values in enumerate(test_loader):
                # print("Values in test_loader:", values)
                inputs, targets = values[:2]  # Change this line
                # print("Inputs:", inputs)
                # print("Targets:", targets)
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predicted == targets).float().sum(dim=1).eq(targets.size(1)).sum().item()
                total += targets.size(0)

        test_acc = 100. * correct / total
        summary_writer.add_scalar('test/loss', test_loss / (batch_idx + 1), epoch)
        summary_writer.add_scalar('test/accuracy', test_acc, epoch)
        print(f'Test Loss: {test_loss / (batch_idx + 1)} | Test Acc: {test_acc}%')





    print('==> Preparing data ..')
    data_dir = os.path.join(cfg.data_dir, "VOC")
        
    train_dataset = create_voc_datasets(data_dir, is_training=True)
    subset_indices = random.sample(range(len(train_dataset)), len(train_dataset) // 5)  # Replace 2 with any other factor
    train_dataset = Subset(train_dataset, subset_indices)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=0,collate_fn=custom_collate,
    )
    # for x in train_loader:

    #     print(x)
        # exit(100)

    eval_dataset = create_voc_datasets(data_dir, is_training=False)
    subset_indices = random.sample(range(len(eval_dataset)), len(eval_dataset) // 5)  # Replace 2 with any other factor
    eval_dataset = Subset(eval_dataset, subset_indices)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=0,collate_fn=custom_collate,
    )
    
    
    print('==> Building ResNet..')
    model = resnet20(wbits=cfg.Wbits, abits=cfg.Abits, num_classes=20).cuda()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.wd)
    lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, [100, 150, 180], gamma=0.1)
    criterion = torch.nn.BCEWithLogitsLoss().cuda()
    summary_writer = SummaryWriter(cfg.log_dir)

    if cfg.pretrain:
        model.load_state_dict(torch.load(cfg.pretrain_dir))
        
    for epoch in range(cfg.max_epochs):
        lr_schedu.step(epoch)
        train(epoch, model, criterion, optimizer, train_loader, summary_writer)
        test(epoch, model, criterion, eval_loader, summary_writer)

        torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, "checkpoint.t7"))

    summary_writer.close()

    # Training
  
def custom_collate(batch):
    inputs = []
    targets = []

    for b in batch:
        inputs.append(b[0])
        target = torch.zeros(len(voc_classes), dtype=torch.float32)
        for obj_class in b[1]['annotation']['object']:
            class_idx = voc_classes.index(obj_class['name'])
            target[class_idx] = 1.0
        targets.append(target)

    inputs = torch.stack(inputs)
    targets = torch.stack(targets)

    return inputs, targets

if __name__ == '__main__':
  mp.set_start_method('spawn', force=True)
  main()
