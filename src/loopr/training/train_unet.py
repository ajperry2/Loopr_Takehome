from tqdm import tqdm
import torch
import numpy as np

from loopr.config.training_nn import TrainingNNConfig
from loopr.losses.dice import dice_coefficient, dice_per_class


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    run_loss, run_dice, n = 0.0, 0.0, 0
    for imgs, masks, _ in tqdm(loader, desc="Train", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, masks)
        loss.backward()
        optimizer.step()
        bs = imgs.size(0); n += bs
        run_loss += loss.item() * bs
        run_dice += dice_coefficient(out, masks).item() * bs
    return {"loss": run_loss/n, "dice": run_dice/n}


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    run_loss, run_dice, n = 0.0, 0.0, 0
    for imgs, masks, _ in tqdm(loader, desc="Valid", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        out = model(imgs)
        loss = criterion(out, masks)
        bs = imgs.size(0); n += bs
        run_loss += loss.item() * bs
        run_dice += dice_coefficient(out, masks).item() * bs
    return {"loss": run_loss/n, "dice": run_dice/n}


class EarlyStopping:
    def __init__(self, patience=5, mode="max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.count = 0
        self.stop = False

    def __call__(self, score):
        if self.best is None:
            self.best = score; self.count = 0
        else:
            improve = (score > self.best) if self.mode=="max" else (score < self.best)
            if improve:
                self.best = score; self.count = 0
            else:
                self.count += 1
                if self.count >= self.patience:
                    self.stop = True


def fit(model, train_loader, val_loader, optimizer, scheduler, criterion, device,
        num_epochs=TrainingNNConfig.epochs, early_stopping_patience=5, save_path=TrainingNNConfig.pretrained_path):
    early_stopping = EarlyStopping(patience=early_stopping_patience, mode="max")
    best_dice = -1.0

    history = {"train": [], "valid": []}

    for epoch in range(1, num_epochs+1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics   = validate_one_epoch(model, val_loader, criterion, device)

        # per-class dice ekle
        all_val_dice = []
        for imgs, masks, _ in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)
            all_val_dice.append(dice_per_class(out, masks))
        val_metrics["per_class_dice"] = np.mean(all_val_dice, axis=0).tolist()

        scheduler.step(val_metrics["dice"])

        print(f"Train Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}")
        print(f"Valid Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}")

        history["train"].append({**train_metrics, "lr": optimizer.param_groups[0]["lr"]})
        history["valid"].append(val_metrics)

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved at epoch {epoch} (dice={best_dice:.4f})")

        early_stopping(val_metrics["dice"])
        if early_stopping.stop:
            print("⏹️ Early stopping triggered.")
            break

    print(f"Training finished. Best Dice = {best_dice:.4f}")
    return history


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    run_loss, run_dice, n = 0.0, 0.0, 0
    for imgs, masks, _ in tqdm(loader, desc="Train", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, masks)
        loss.backward()
        optimizer.step()
        bs = imgs.size(0); n += bs
        run_loss += loss.item() * bs
        run_dice += dice_coefficient(out, masks).item() * bs
    return {"loss": run_loss/n, "dice": run_dice/n}


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    run_loss, run_dice, n = 0.0, 0.0, 0
    for imgs, masks, _ in tqdm(loader, desc="Valid", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        out = model(imgs)
        loss = criterion(out, masks)
        bs = imgs.size(0); n += bs
        run_loss += loss.item() * bs
        run_dice += dice_coefficient(out, masks).item() * bs
    return {"loss": run_loss/n, "dice": run_dice/n}


class EarlyStopping:
    def __init__(self, patience=5, mode="max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.count = 0
        self.stop = False

    def __call__(self, score):
        if self.best is None:
            self.best = score; self.count = 0
        else:
            improve = (score > self.best) if self.mode=="max" else (score < self.best)
            if improve:
                self.best = score; self.count = 0
            else:
                self.count += 1
                if self.count >= self.patience:
                    self.stop = True


def fit(model, train_loader, val_loader, optimizer, scheduler, criterion, device,
        num_epochs=TrainingNNConfig.epochs, early_stopping_patience=TrainingNNConfig.early_stopping_patience, save_path=TrainingNNConfig.pretrained_path):
    early_stopping = EarlyStopping(patience=early_stopping_patience, mode="max")
    best_dice = -1.0

    history = {"train": [], "valid": []}

    for epoch in range(1, num_epochs+1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics   = validate_one_epoch(model, val_loader, criterion, device)

        # per-class dice ekle
        all_val_dice = []
        for imgs, masks, _ in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)
            all_val_dice.append(dice_per_class(out, masks))
        val_metrics["per_class_dice"] = np.mean(all_val_dice, axis=0).tolist()

        scheduler.step(val_metrics["dice"])

        print(f"Train Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}")
        print(f"Valid Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}")

        history["train"].append({**train_metrics, "lr": optimizer.param_groups[0]["lr"]})
        history["valid"].append(val_metrics)

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved at epoch {epoch} (dice={best_dice:.4f})")

        early_stopping(val_metrics["dice"])
        if early_stopping.stop:
            print("⏹️ Early stopping triggered.")
            break

    print(f"Training finished. Best Dice = {best_dice:.4f}")
    return history
    