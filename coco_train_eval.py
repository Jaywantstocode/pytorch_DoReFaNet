import os
import time
import argparse
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json


cudnn.benchmark = True
import torchvision

from tensorboardX import SummaryWriter

from nets.coco_resnet import *

from utils.preprocessing import *

# Training settings
parser = argparse.ArgumentParser(description='DoReFa-Net pytorch implementation')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_name', type=str, default='resnet_w1a32')
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--pretrain_dir', type=str, default='./ckpt/resnet20_baseline')

parser.add_argument('--cifar', type=int, default=10)

parser.add_argument('--Wbits', type=int, default=1)
parser.add_argument('--Abits', type=int, default=32)

parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--wd', type=float, default=1e-4)

parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--eval_batch_size', type=int, default=100)
parser.add_argument('--max_epochs', type=int, default=200)

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


def main():
  if cfg.cifar == 10:
    print('training CIFAR-10 !')
    dataset = torchvision.datasets.CIFAR10
  elif cfg.cifar == 100:
    print('training CIFAR-100 !')
    dataset = torchvision.datasets.CIFAR100
  else:
    assert False, 'dataset unknown !'


  print('==> Preparing data ..')
  train_dataset = CocoDetection(root=os.path.join(cfg.data_dir, 'train2017'),
                                annFile=os.path.join(cfg.data_dir, 'annotations/instances_train2017.json'),
                                transform=coco_transform(is_training=True))
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True,
                                            num_workers=cfg.num_workers)

  eval_dataset = CocoDetection(root=os.path.join(cfg.data_dir, 'val2017'),
                              annFile=os.path.join(cfg.data_dir, 'annotations/instances_val2017.json'),
                              transform=coco_transform(is_training=False))
  eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg.eval_batch_size, shuffle=False,
                                            num_workers=cfg.num_workers)
  
  
  print('==> Building ResNet..')
  
  model = resnet20(wbits=cfg.Wbits, abits=cfg.Abits).cuda()

  optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.wd)
  lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, [100, 150, 180], gamma=0.1)
  criterion = torch.nn.MSELoss().cuda()
  summary_writer = SummaryWriter(cfg.log_dir)

  if cfg.pretrain:
    model.load_state_dict(torch.load(cfg.pretrain_dir))

  # Training
  def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()

    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
      inputs = inputs.cuda()
      targets = torch.stack([torch.tensor(ann['bbox']) for ann in annotations]).cuda()

      outputs = model(inputs)
      loss = criterion(outputs, targets)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if batch_idx % cfg.log_interval == 0:
        step = len(train_loader) * epoch + batch_idx
        duration = time.time() - start_time

        print('%s epoch: %d step: %d cls_loss= %.5f (%d samples/sec)' %
              (datetime.now(), epoch, batch_idx, loss.item(),
               cfg.train_batch_size * cfg.log_interval / duration))

        start_time = time.time()
        summary_writer.add_scalar('cls_loss', loss.item(), step)
        summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], step)

  def test(epoch):
    model.eval()

    results = []
    img_ids = []
    for batch_idx, (inputs, annotations) in enumerate(eval_loader):
        inputs = inputs.cuda()

        outputs = model(inputs)

        for i, output in enumerate(outputs):
            results.append({
                'image_id': int(annotations[i]['image_id']),
                'category_id': int(output['category_id']),
                'bbox': output['bbox'].tolist(),
                'score': float(output['score'])
            })
            img_ids.append(int(annotations[i]['image_id']))

      # Save the results to a JSON file
    with open(os.path.join(cfg.ckpt_dir, 'predictions.json'), 'w') as f:
        json.dump(results, f)

    # Evaluate using COCO API
    coco_gt = COCO(os.path.join(cfg.data_dir, 'annotations/instances_val2017.json'))
    coco_dt = coco_gt.loadRes(os.path.join(cfg.ckpt_dir, 'predictions.json'))
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    summary_writer.add_scalar('AP@0.5', coco_eval.stats[1], global_step=epoch)


if __name__ == '__main__':
  main()
