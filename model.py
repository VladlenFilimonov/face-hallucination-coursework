import torch
from torch import nn
import torch.optim as optim
from loss import GeneratorLoss


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.input_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.res_block1 = ResidualBlock(64)
        self.res_block2 = ResidualBlock(64)
        self.res_block3 = ResidualBlock(64)
        self.res_block4 = ResidualBlock(64)
        self.res_block5 = ResidualBlock(64)
        self.res_blocks_end = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

        self.up_sample_block1 = UpsampleBLock(64, 2)
        self.up_sample_block2 = UpsampleBLock(64, 2)
        self.output = nn.Conv2d(64, 3, kernel_size=9, padding=4)
        self.output_block = nn.Sequential(self.up_sample_block1,
                                          self.up_sample_block2,
                                          self.output)

    def forward(self, x):
        input_block = self.input_block(x)
        res_blocks_network = self.res_block1(input_block)
        res_blocks_network = self.res_block2(res_blocks_network)
        res_blocks_network = self.res_block3(res_blocks_network)
        res_blocks_network = self.res_block4(res_blocks_network)
        res_blocks_network = self.res_block5(res_blocks_network)
        res_blocks_end = self.res_blocks_end(res_blocks_network)
        skip_connection = input_block + res_blocks_end
        output_block = self.output_block(skip_connection)

        shift_to_positives = torch.tanh(output_block) + 1
        scale_to_normal = shift_to_positives / 2
        return scale_to_normal


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class SrganNetwork:
    def __init__(self):
        self.net_g = Generator()
        print('# generator parameters:', sum(param.numel() for param in self.net_g.parameters()))
        self.net_d = Discriminator()
        print('# discriminator parameters:', sum(param.numel() for param in self.net_d.parameters()))

        self.loss_function = GeneratorLoss()

        if torch.cuda.is_available():
            self.net_g.cuda()
            self.net_d.cuda()
            self.loss_function.cuda()

        self.optimizer_g = optim.Adam(self.net_g.parameters())
        self.optimizer_d = optim.Adam(self.net_d.parameters())

