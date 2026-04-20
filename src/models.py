import torch
import torch.nn as nn
import random


# ---- ResNet Generator (Paired CycleGAN) ----

class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, features=64, num_blocks=9):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, features, 7, 1, 0, bias=False),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
        ]
        for i in range(2):
            in_f  = features * (2 ** i)
            out_f = features * (2 ** (i + 1))
            layers += [
                nn.Conv2d(in_f, out_f, 3, 2, 1, bias=False),
                nn.InstanceNorm2d(out_f),
                nn.ReLU(inplace=True),
            ]
        for _ in range(num_blocks):
            layers.append(ResNetBlock(features * 4))
        for i in range(2):
            in_f  = features * (4 // (2 ** i))
            out_f = features * (4 // (2 ** (i + 1)))
            layers += [
                nn.ConvTranspose2d(in_f, out_f, 3, 2, 1, output_padding=1, bias=False),
                nn.InstanceNorm2d(out_f),
                nn.ReLU(inplace=True),
            ]
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(features, out_ch, 7, 1, 0),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ---- U-Net Generator (Pix2Pix) ----

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, use_bn=True, dropout=False):
        super().__init__()
        if down:
            self.conv = nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)
        else:
            self.conv = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.act  = nn.LeakyReLU(0.2, inplace=True) if down else nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.5) if dropout else nn.Identity()

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x))))


class UNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, features=64):
        super().__init__()
        self.e1     = nn.Conv2d(in_ch, features, 4, 2, 1, bias=False)
        self.e2     = UNetBlock(features,     features * 2, down=True)
        self.e3     = UNetBlock(features * 2, features * 4, down=True)
        self.e4     = UNetBlock(features * 4, features * 8, down=True)
        self.e5     = UNetBlock(features * 8, features * 8, down=True)
        self.e6     = UNetBlock(features * 8, features * 8, down=True)
        self.e7     = UNetBlock(features * 8, features * 8, down=True)
        self.e8     = UNetBlock(features * 8, features * 8, down=True, use_bn=False)

        self.d1     = UNetBlock(features * 8,     features * 8, down=False, dropout=True)
        self.d2     = UNetBlock(features * 8 * 2, features * 8, down=False, dropout=True)
        self.d3     = UNetBlock(features * 8 * 2, features * 8, down=False, dropout=True)
        self.d4     = UNetBlock(features * 8 * 2, features * 8, down=False)
        self.d5     = UNetBlock(features * 8 * 2, features * 4, down=False)
        self.d6     = UNetBlock(features * 4 * 2, features * 2, down=False)
        self.d7     = UNetBlock(features * 2 * 2, features,     down=False)
        self.out    = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_ch, 4, 2, 1),
            nn.Tanh()
        )
        self.act_e1 = nn.LeakyReLU(0.2, inplace=True)

        # ── Attention gates on skip connections ──────────────────────
        self.att1 = AttentionGate(features * 8)
        self.att2 = AttentionGate(features * 8)
        self.att3 = AttentionGate(features * 8)
        self.att4 = AttentionGate(features * 8)
        self.att5 = AttentionGate(features * 4)
        self.att6 = AttentionGate(features * 2)
        self.att7 = AttentionGate(features)

    def forward(self, x):
        e1 = self.act_e1(self.e1(x))
        e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3)
        e5 = self.e5(e4); e6 = self.e6(e5); e7 = self.e7(e6)
        e8 = self.e8(e7)

        d1 = self.d1(e8)
        d2 = self.d2(torch.cat([d1, self.att1(d1, e7)], dim=1))
        d3 = self.d3(torch.cat([d2, self.att2(d2, e6)], dim=1))
        d4 = self.d4(torch.cat([d3, self.att3(d3, e5)], dim=1))
        d5 = self.d5(torch.cat([d4, self.att4(d4, e4)], dim=1))
        d6 = self.d6(torch.cat([d5, self.att5(d5, e3)], dim=1))
        d7 = self.d7(torch.cat([d6, self.att6(d6, e2)], dim=1))
        return self.out(torch.cat([d7, self.att7(d7, e1)], dim=1))



# ---- PatchGAN Discriminator ----

class PatchGANDiscriminator(nn.Module):
    """
    Single-input PatchGAN for CycleGAN variants.
    Set in_ch=6 for Pix2Pix (concatenated input+target).
    Set in_ch=3 for CycleGAN (single modality).
    """
    def __init__(self, in_ch=3, features=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features,     features * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 4, features * 8, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 8, 1, 4, 1, 1, bias=False),
        )

    def forward(self, x):
        return self.model(x)


# ---- utilities ----

def init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            if m.weight is not None:
                nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)


class ImageBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.buffer   = []

    def push_and_pop(self, images):
        result = []
        for image in images:
            image = image.unsqueeze(0)
            if len(self.buffer) < self.max_size:
                self.buffer.append(image)
                result.append(image)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    tmp = self.buffer[idx].clone()
                    self.buffer[idx] = image
                    result.append(tmp)
                else:
                    result.append(image)
        return torch.cat(result, dim=0)



class AttentionGate(nn.Module):
    """
    Soft attention on skip connections.
    Helps generator focus on ambiguous soft tissue regions for CT->MR.
    """
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x, skip):
        attention = self.gate(torch.cat([x, skip], dim=1))
        return skip * attention



class MultiScaleDiscriminator(nn.Module):
    """
    Two PatchGAN discriminators operating at different scales.
    Scale 1: original resolution
    Scale 2: 2x downsampled
    Helps CT->MR capture both coarse tissue boundaries and fine texture.
    """
    def __init__(self, in_ch=3, features=64):
        super().__init__()
        self.D1 = PatchGANDiscriminator(in_ch=in_ch, features=features)
        self.D2 = PatchGANDiscriminator(in_ch=in_ch, features=features)
        self.downsample = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        pred1 = self.D1(x)
        pred2 = self.D2(self.downsample(x))
        return pred1, pred2