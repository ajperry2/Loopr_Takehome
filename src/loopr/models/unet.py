from torch import nn
from torchvision.models import ResNet18_Weights
import torch
import torchvision.models as models
from typing import Optional
from pathlib import Path
import torch.nn.functional as F

class UNetResNet18(nn.Module):
    def __init__(
        self, 
        num_classes=4, 
        pretrained=False, 
        decoder_mode="add", 
        dropout=0.0
    ):
        super().__init__()
        # Encoder backbone
        base = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        del base.fc, base.avgpool  # UNet'te kullanılmıyor

        self.enc1 = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.enc2 = nn.Sequential(base.maxpool, base.layer1)
        self.enc3 = base.layer2
        self.enc4 = base.layer3
        self.enc5 = base.layer4

        self.mode = decoder_mode

        def up_block(in_ch, out_ch, use_concat=False):
            layers = [nn.ConvTranspose2d(in_ch, out_ch, 2, 2),
                      nn.BatchNorm2d(out_ch),
                      nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout2d(p=dropout))
            if use_concat:
                layers += [nn.Conv2d(out_ch*2, out_ch, 3, padding=1),
                           nn.BatchNorm2d(out_ch),
                           nn.ReLU(inplace=True)]
            return nn.Sequential(*layers)

        if self.mode == "add":
            self.up4 = up_block(512,256)
            self.up3 = up_block(256,128)
            self.up2 = up_block(128,64)
            self.up1 = up_block(64,64)
        else:  # concat
            self.up4 = up_block(512,256, use_concat=True)
            self.up3 = up_block(256,128, use_concat=True)
            self.up2 = up_block(128,64, use_concat=True)
            self.up1 = up_block(64,64, use_concat=True)

        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        # Decoder
        if self.mode == "add":
            d4 = self.up4(e5) + e4
            d3 = self.up3(d4) + e3
            d2 = self.up2(d3) + e2
            d1 = self.up1(d2) + e1
        else:  # concat
            d4 = self.up4(torch.cat([F.interpolate(e5, size=e4.shape[2:], mode="bilinear", align_corners=False), e4],1))
            d3 = self.up3(torch.cat([F.interpolate(d4, size=e3.shape[2:], mode="bilinear", align_corners=False), e3],1))
            d2 = self.up2(torch.cat([F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False), e2],1))
            d1 = self.up1(torch.cat([F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False), e1],1))

        out = self.final(d1)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out
        
    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        return self.enc5(x)

def load_unet(path: Optional[Path] = None, gpu=True):
    unet_model = UNetResNet18()
    if path is not None and path.exists():
        model_dict = torch.load(path, weights_only=False, map_location=torch.device('cpu'))
        unet_model.load_state_dict(model_dict)
        unet_model.eval()
    if gpu:
        unet_model = unet_model.cuda()
    return unet_model