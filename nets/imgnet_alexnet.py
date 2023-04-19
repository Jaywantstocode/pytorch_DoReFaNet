import torch
import torch.nn as nn
import torch.nn.init as init

from utils.quant_dorefa import *
import torchvision.models
import torchvision.transforms as transforms

class AlexNet_Q(torchvision.models.AlexNet):
  def __init__(self, wbit, abit, num_classes=1000):
    super(AlexNet_Q, self).__init__(num_classes=num_classes)

    self.wbit = wbit
    self.abit = abit
    Conv2d_Q = conv2d_Q_fn(wbit)
    self.features = nn.Sequential(
        Conv2d_Q(3, 64, kernel_size=11, stride=4, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        Conv2d_Q(64, 192, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        Conv2d_Q(192, 384, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        Conv2d_Q(384, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        Conv2d_Q(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
    )
    Linear_Q = linear_Q_fn(wbit)
    self.classifier = nn.Sequential(
        Linear_Q(256 * 6 * 6, 4096),
        nn.ReLU(inplace=True),
        activation_quantize_fn(abit),
        nn.Dropout(),
        Linear_Q(4096, 4096),
        nn.ReLU(inplace=True),
        activation_quantize_fn(abit),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), 256 * 6 * 6)
    x = self.classifier(x)
    return x


if __name__ == '__main__':
  from torch.autograd import Variable

  features = []


  def hook(self, input, output):
    print(output.data.cpu().numpy().shape)
    features.append(output.data.cpu().numpy())


  net = AlexNet_Q(wbit=1, abit=2)
  net.train()

  for w in net.named_parameters():
    print(w[0])

  for m in net.modules():
    m.register_forward_hook(hook)

  y = net(Variable(torch.randn(1, 3, 224, 224)))
  print(y.size())
