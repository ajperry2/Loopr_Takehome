from torch import nn
from loopr.models.unet import load_unet

from loopr.config.training_nn import TrainingNNConfig

class Model(nn.Module):
    def __init__(self):
        unet_model = load_unet(TrainingNNConfig.pretrained_path,gpu=False)
        
    def forward(self,x):
        return x