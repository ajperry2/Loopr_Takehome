from torch import nn
from loopr.models.unet import load_unet
from loopr.models.mlp import create_mlp
from loopr.config.training_nn import TrainingNNConfig
import torch

from os import getcwd


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet_model = load_unet(getcwd() / TrainingNNConfig.pretrained_path,gpu=True)
        self.unet_model.eval()
        self.mlp = create_mlp(getcwd() / TrainingNNConfig.pretrained_mlp_path,gpu=True)
        self.mlp.eval()
        # self.linear_layer = list(self.mlp.children())[0].cuda()
        # self.centroids = torch.load(getcwd() / TrainingNNConfig.centroid_file)
        
    def forward(self, image_tiles):
        embeddings = self.unet_model.encode(image_tiles.cuda())
        embeddings = embeddings.reshape(len(image_tiles),-1)
        logits = self.mlp(embeddings)
        votes = logits.argmax(dim=1)
        if (votes!=0).any():
            for vote in votes:
                if vote != 0:
                    return {"prediction": vote.item(), "logits": logits.cpu().detach().numpy().tolist()}
        else:
            return {"prediction": 0, "logits": logits.cpu().detach().numpy().tolist()}