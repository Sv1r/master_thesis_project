import segmentation_models_pytorch
import torch
# import torch.nn as nn

import settings


class Discriminator(torch.nn.Module):
    """Discriminator Network"""
    def __init__(self):
        super(Discriminator, self).__init__()

        self.in_channels = settings.NUMBER_OF_INPUT_CHANNELS
        self.ndf = 64
        self.out_channels = settings.NUMBER_OF_CLASSES

        self.main = torch.nn.Sequential(
            torch.nn.Conv2d((self.in_channels + self.out_channels), self.ndf, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.InstanceNorm2d(self.ndf * 2),
            torch.nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.InstanceNorm2d(self.ndf * 4),
            torch.nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 1, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.InstanceNorm2d(self.ndf * 8),
            torch.nn.Conv2d(self.ndf * 8, self.out_channels, 4, 1, 1, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x, label):
        out = torch.cat((x, label), dim=1)
        out = self.main(out)
        return out


generator = segmentation_models_pytorch.UnetPlusPlus(
    encoder_name='mobilenet_v2',
    encoder_weights='imagenet',
    in_channels=settings.NUMBER_OF_INPUT_CHANNELS,
    classes=settings.NUMBER_OF_CLASSES,
    activation='sigmoid'
)
