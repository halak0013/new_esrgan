import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_act, **kwargs):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=True)
        self.act = (
            nn.LeakyReLU(negative_slope=0.2, inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.cnn(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, stride=1, bias=True
        )
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(self.upsample(x)))


class DenseResidualBlock(nn.Module):
    def __init__(self, in_channels, channels=32, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.blocks = nn.ModuleList()

        for i in range(5):
            self.blocks.append(
                ConvBlock(
                    in_channels + channels * i,
                    channels if i <= 3 else in_channels,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    use_act=True if i <= 3 else False,
                )
            )

    def forward(self, x):
        new_input = x
        for block in self.blocks:
            out = block(new_input)
            new_input = torch.cat([new_input, out], dim=1)
        return self.residual_beta * out + x


class RRDB(nn.Module):
    def __init__(self, in_channels, channels=32, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.rrdb = nn.Sequential(*[DenseResidualBlock(in_channels) for _ in range(3)])

    def forward(self, x):
        return self.residual_beta * self.rrdb(x) + x


class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=23):
        super().__init__()

        self.initial_conv = nn.Conv2d(
            in_channels, num_channels, kernel_size=3, padding=1, stride=1, bias=True
        )

        self.rrdb_blocks = nn.Sequential(
            *[RRDB(num_channels) for _ in range(num_blocks)]
        )
        self.conv = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, padding=1, stride=1, bias=True
        )
        self.upsample_blocks = nn.Sequential(
            UpsampleBlock(num_channels),
            UpsampleBlock(num_channels),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(
                num_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=True,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                num_channels, in_channels, kernel_size=3, padding=1, stride=1, bias=True
            ),
        )

    def forward(self, x):
        initial = self.initial_conv(x)
        x = self.conv(self.rrdb_blocks(initial)) + initial
        x = self.upsample_blocks(x)
        return self.final_conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride=1 + idx % 2,
                    padding=1,
                    use_act=True,
                ),
            )
            in_channels = feature

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)


def initialize_weights(model, scale=0.1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale


# def initialize_weights_modern(model, scale=0.1, use_spectral_norm=True):
#     """Modern hybrid initialization approach"""
#     import torch.nn.utils.spectral_norm as spectral_norm
#     for name, m in model.named_modules():
#         if isinstance(m, nn.Conv2d):
#             # Spectral normalization (opsiyonel)
#             if use_spectral_norm and 'discriminator' in name.lower():
#                 spectral_norm(m)

#             # Kaiming initialization
#             nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='leaky_relu')
#             m.weight.data *= scale

#             # Bias initialization
#             if m.bias is not None:
#                 nn.init.constant_(m.bias.data, 0.0)

#         elif isinstance(m, nn.Linear):
#             if use_spectral_norm:
#                 spectral_norm(m)

#             nn.init.kaiming_normal_(m.weight.data)
#             m.weight.data *= scale

#             if m.bias is not None:
#                 nn.init.constant_(m.bias.data, 0.0)

#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.constant_(m.weight.data, 1.0)
#             nn.init.constant_(m.bias.data, 0.0)


def test():
    gen = Generator()
    disc = Discriminator()
    low_res = 24
    x = torch.randn((5, 3, low_res, low_res))
    gen_out = gen(x)
    disc_out = disc(gen_out)

    print(gen_out.shape)
    print(disc_out.shape)


if __name__ == "__main__":
    test()
