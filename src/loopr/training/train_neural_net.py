import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from sklearn.metrics import f1_score 
import matplotlib.pyplot as plt
import numpy as np

from loopr.models.mlp import create_mlp
from loopr.models.unet import load_unet
from loopr.data.classification import (
    ClassificationDataset, 
    collate_fn, 
    get_train_transforms, 
    get_valid_transforms )
from loopr.config.training_nn import TrainingNNConfig

def train_neural_net():
    # Load Models
    # UNet
    unet_model = load_unet(
        TrainingNNConfig.pretrained_path, 
        gpu=True
    )
    # MLP
    mlp = create_mlp(TrainingNNConfig.pretrained_mlp_path, gpu=True)
    
    # Data Loading
    train_ds =  ClassificationDataset(
        transforms=get_train_transforms(), 
        training=True,
        train_split=TrainingNNConfig.train_split,
        censor_files=True
    )
    val_ds =  ClassificationDataset(
        transforms=get_valid_transforms(), 
        training=False,
        train_split=TrainingNNConfig.train_split,
        censor_files=True
    )
    train_loader   = DataLoader(
        train_ds,   
        batch_size=TrainingNNConfig.batch_size, 
        shuffle=True,      
        num_workers=TrainingNNConfig.num_workers, 
        pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=TrainingNNConfig.batch_size, shuffle=False,
                              num_workers=TrainingNNConfig.num_workers, pin_memory=True)
    # Loss / Opt.
    pos_weight = torch.ones([4]).cuda()  # All weights are equal to 1

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
    # History
    val_scores = []
    val_losses = []
    training_losses = []
    best_score = 0
    for epoch in range(TrainingNNConfig.epochs):
        # Run through training:
        print(f"epoch: {epoch}")
        train_loss_total = 0
        train_num_batches = 0
        mlp.train()
        for images, dense_classes, metas in train_loader:
            classes = torch.nn.functional.one_hot(dense_classes, num_classes=4).float().to('cuda')
            embeddings = unet_model.encode(images.cuda())
            embeddings = embeddings.reshape(len(embeddings), -1)
            optimizer.zero_grad()
            domain_embedded = mlp(embeddings.cuda())
            loss = criterion(domain_embedded, classes)
            train_loss_total += loss.sum().item()
            train_num_batches += 1
            
            loss.backward()
            optimizer.step()
        # validate
        training_losses.append(train_loss_total/train_num_batches)
        mlp.eval()
        val_score = 0
        val_loss = 0
        num_val_batches = 0
        for images, dense_classes, metas in val_loader:
            classes = torch.nn.functional.one_hot(dense_classes, num_classes=4).float().to('cuda')
            embeddings = unet_model.encode(images.cuda())
            embeddings = embeddings.reshape(len(embeddings), -1)
            domain_embedded = mlp(embeddings.cuda())
            loss = criterion(domain_embedded, classes)
            val_loss += loss.sum().item()
            domain_embedded = (domain_embedded > 0.5) 
            score = f1_score(
                torch.nn.functional.one_hot(dense_classes, num_classes=4).cpu().detach(), 
                domain_embedded.cpu().detach(),
                average='micro'
            )
            num_val_batches += 1
            val_score += score
        val_losses.append(val_loss/ num_val_batches)
        val_scores.append(val_score / num_val_batches)
        print(f"Train Loss (BCE + lambda * L2): {train_loss_total / train_num_batches}")
        print(f"Validation Loss (BCE + lambda * L2): {val_loss/ num_val_batches}")
        print(f"Validation Score (F1 micro score): {val_score / num_val_batches}")
        mlp.train()
        if val_score / num_val_batches > best_score:
            best_score = val_score / num_val_batches
            torch.save(mlp, TrainingNNConfig.pretrained_mlp_path)
    # Plot history
    plt.plot(np.arange(len(val_losses)), val_losses, label="Validation")
    plt.plot(np.arange(len(training_losses)), training_losses, label="Training", color="orangered")
    plt.title("Losses MLP")
    plt.legend()

    
    plt.plot(np.arange(len(val_scores)), val_scores, label="Validation", color="blue")
    plt.title("Micro F1 Score MLP")
    plt.legend()