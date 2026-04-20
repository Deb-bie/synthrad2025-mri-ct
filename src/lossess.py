import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """
    LSGAN loss — uses MSE instead of BCE for stable training.
    Creates labels dynamically from actual discriminator output shape
    to avoid hardcoded shape errors.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def __call__(self, prediction, is_real):
        if is_real:
            target = torch.ones_like(prediction)
        else:
            target = torch.zeros_like(prediction)
        return self.loss(prediction, target)


class PairedL1Loss(nn.Module):
    """
    Masked L1 loss — only penalizes errors inside the body mask.
    Prevents background from dominating the loss.
    """
    def __init__(self, lambda_weight=10.0):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.loss          = nn.L1Loss()

    def __call__(self, prediction, target, mask):
        return self.loss(prediction * mask, target * mask) * self.lambda_weight


class CycleConsistencyLoss(nn.Module):
    """
    Cycle consistency loss — reconstructed image should match original.
    F(G(MR)) ≈ MR and G(F(CT)) ≈ CT
    """
    def __init__(self, lambda_weight=10.0):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.loss          = nn.L1Loss()

    def __call__(self, reconstructed, original, mask):
        return self.loss(reconstructed * mask, original * mask) * self.lambda_weight


class IdentityLoss(nn.Module):
    """
    Identity loss — G(CT) should give CT, F(MR) should give MR.
    Prevents generators from making unnecessary changes.
    """
    def __init__(self, lambda_weight=5.0):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.loss          = nn.L1Loss()

    def __call__(self, identity_output, real_input, mask):
        return self.loss(identity_output * mask, real_input * mask) * self.lambda_weight



class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(pretrained=True).features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg   = vgg
        self.l1    = nn.L1Loss()

    def forward(self, fake, real, mask):
        fake_m  = fake * mask
        real_m  = real * mask
        # VGG expects 3 channels
        fake_3  = fake_m.repeat(1, 3, 1, 1) if fake_m.shape[1] != 3 else fake_m
        real_3  = real_m.repeat(1, 3, 1, 1) if real_m.shape[1] != 3 else real_m
        return self.l1(self.vgg(fake_3), self.vgg(real_3))

