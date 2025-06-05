import torch
import torch.nn as nn
import torchvision.models as models
import src.utils.config as cfg


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Fix: Use weights parameter instead of pretrained
        self.vgg = (
            models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            .features[:36]
            .eval()
            .to(cfg.DEVICE)
        )
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            # parametre güncellemesini kapat
            param.requires_grad = False

        # VGG'yi compile edin (PyTorch 2.0+)
        # if hasattr(torch, 'compile'):
        #     self.vgg = torch.compile(self.vgg)

    def forward(self, input, target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)


# def forward(self, input, target):
#     vgg_input_features = self.vgg(input)
#     vgg_target_features = self.vgg(target)

#     # Birden fazla loss kombinasyonu
#     mse_loss = nn.MSELoss()(vgg_input_features, vgg_target_features)
#     l1_loss = nn.L1Loss()(vgg_input_features, vgg_target_features)

#     return 0.8 * mse_loss + 0.2 * l1_loss
