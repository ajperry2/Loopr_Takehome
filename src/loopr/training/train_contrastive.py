from torch import nn
import torch
from collections import defaultdict
from sklearn.manifold import TSNE
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from loopr.models.unet import load_unet
from loopr.models.mlp import create_mlp
from loopr.losses.contrastive import ContrastiveLoss

def train_contrastive(censored_val_loader: DataLoader):
    # Load Models
    unet_model = load_unet(TrainingNNConfig.pretrained_path,gpu=True)
    mlp = create_mlp(TrainingNNConfig.pretrained_mlp_path, gpu=True)
    linear_layer = list(mlp.children())[0]

    # Loss / Opt.
    pos_weight = torch.ones([4]).cuda()  # All weights are equal to 1
    pos_weight[0] = 95/141
    pos_weight[1] = 7/141
    pos_weight[2] = 30/141
    pos_weight[3] = 9/141
    criterion = ContrastiveLoss()
    optimizer = Adam(
        mlp.parameters(), 
        lr=TrainingNNConfig.lr, 
        weight_decay=TrainingNNConfig.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode="max", 
        factor=0.5, 
        patience=3
    )
    
    training_losses = []
    best_score = 0
    
    for epoch in range(TrainingNNConfig.epochs):
        # Run through training:
        print(f"epoch: {epoch}")
        train_loss_total = 0
        train_num_batches = 0
        for images, dense_classes, metas in censored_val_loader:
            if len(images) % 2 != 0:
                continue
                
            embeddings = unet_model.encode(images.cuda())
            embeddings = embeddings.reshape(len(embeddings), -1)
            domain_embedded = linear_layer(embeddings.cuda())
            optimizer.zero_grad()
            x_1 = domain_embedded[0::2]
            x_2 = domain_embedded[1::2]
            label = (dense_classes[0::2] != dense_classes[1::2]).float().cuda()
            loss = criterion(x_1,x_2,label)
            training_losses.append(loss.sum().item())
            train_num_batches += 1
            
            loss.backward()
            optimizer.step()
        # validate
        # mlp.eval()
        val_score = 0
        val_loss = 0
        # num_val_batches = 0
    
    
        # val_losses.append(val_loss/ num_val_batches)
        # val_scores.append(val_score / num_val_batches)
        print(f"Train Loss (Contrastive): {training_losses[-1]}")
        # print(f"Validation Score (F1 micro score): {val_score / num_val_batches}")
        # mlp.train()
        if training_losses[-1] > best_score:
            best_score = train_loss_total / train_num_batches
            torch.save(linear_layer, TrainingNNConfig.pretrained_contrastive_path)