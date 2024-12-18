import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
  def __init__(self, kernelSize = 3, inChannels = 128, outChannels = 128, strd = 1, paddng = 1):
    super().__init__()
    self.block = nn.Sequential(
        nn.Conv2d(in_channels = inChannels, out_channels = outChannels, kernel_size = kernelSize, stride = strd, padding = paddng),
        nn.BatchNorm2d(128),
        nn.PReLU(),
        nn.Conv2d(in_channels = inChannels, out_channels = outChannels, kernel_size = kernelSize, stride = strd, padding = paddng),
        nn.BatchNorm2d(128)
    )
  def forward(self, x):
    out = self.block(x)
    return torch.add(out, x)
  
class UpsampleBlock(nn.Module):
  def __init__(self, inChannels,scaleFactor):
    super().__init__()
    self.conv = nn.Conv2d(in_channels= inChannels, out_channels= inChannels * scaleFactor ** 2, kernel_size=3, stride=1, padding=1)
    self.ps = nn.PixelShuffle(scaleFactor)
    self.act = nn.PReLU(inChannels)
  def forward(self, x):
    return self.act(self.ps(self.conv(x)))

class SRResnet(nn.Module):
  def __init__(self):
    super(SRResnet, self).__init__()
    self.name = "SRResNet"

    self.l1 = nn.Conv2d(kernel_size=9, stride=1, in_channels=3, out_channels=128, padding=4)
    self.l2 = nn.PReLU()

    self.residuals = nn.Sequential()
    for _ in range(0, 16):
        self.residuals.add_module('residualBlock',ResidualBlock())

    self.postResiduals = nn.Sequential(
        nn.Conv2d(in_channels= 128, out_channels=128, kernel_size= 3, stride=1, padding=1),
        nn.BatchNorm2d(128),
    )
    self.upsample = UpsampleBlock(128, 2)

    self.final = nn.Conv2d(in_channels= 128, out_channels = 3, kernel_size= 9, stride=1, padding=4)

  def forward(self, x):
    x = self.l1(x)
    x1 = self.l2(x)
    x = self.residuals(x1)
    x = self.postResiduals(x)
    x = torch.add(x, x1)
    x = self.upsample(x)
    x = self.final(x)

    return x