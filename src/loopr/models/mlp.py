import torchvision
import torch
from pathlib import Path
from typing import Optional
from loopr.config.training_nn import TrainingNNConfig

def create_mlp(path: Optional[Path] = None, gpu=False):
    if path is not None and path.exists():
        mlp = torch.load(
            path, 
            weights_only=False,
            map_location=torch.device('cpu')
        )
    else:
        mlp = torchvision.ops.MLP(
            in_channels=32768, 
            hidden_channels=[128, 4], 
            activation_layer=torch.nn.modules.activation.ReLU, 
            bias= True, dropout= 0.0)
    if gpu:
        mlp = mlp.cuda()
    return mlp
