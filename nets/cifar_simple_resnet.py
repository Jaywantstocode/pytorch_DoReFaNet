import torch
import torch.nn as nn
import torch.nn.functional as F

class PreActBlock_conv(nn.Module):
    '''Pre-activation version of the BasicBlock.'''

    def __init__(self, in_planes, out_planes, stride=1):
        super(PreActBlock_conv, self).__init__()

        self.bn0 = nn.BatchNorm2d(in_planes)
        self.conv0 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv1 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.skip_conv = None
        if stride != 1:
            self.skip_conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.skip_bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn0(x))

        if self.skip_conv is not None:
            shortcut = self.skip_conv(out)
            shortcut = self.skip_bn(shortcut)
        else:
            shortcut = x

        out = self.conv0(out)
        out = F.relu(self.bn1(out))
        out = self.conv1(out)
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_units, num_classes):
        super(PreActResNet, self).__init__()
        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        self.layers = nn.ModuleList()
        in_planes = 16
        strides = [1] * (num_units[0]) + \
                  [2] + [1] * (num_units[1] - 1) + \
                  [2] + [1] * (num_units[2] - 1)
        channels = [16] * num_units[0] + [32] * num_units[1] + [64] * num_units[2]
        for stride, channel in zip(strides, channels):
            self.layers.append(block(in_planes, channel, stride))
            in_planes = channel

        self.bn = nn.BatchNorm2d(64)
        self.logit = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.conv0(x)
        for layer in self.layers:
            out = layer(out)
        out = self.bn(out)
        out = out.mean(dim=2).mean(dim=2)
        out = self.logit(out)
        return out

def resnet20(num_classes=10):
    return PreActResNet(PreActBlock_conv, [3, 3, 3], num_classes=num_classes)

def resnet56(num_classes=10):
    return PreActResNet(PreActBlock_conv, [9, 9, 9], num_classes=num_classes)

if __name__ == 'main':
  
  features = []
  def hook(self, input, output):
      print(output.data.cpu().numpy().shape)
      features.append(output.data.cpu().numpy())

  net = resnet20(num_classes=10)
  for m in net.modules():
      m.register_forward_hook(hook)

  y = net(torch.randn(1, 3, 32, 32))
  print(y.size())
