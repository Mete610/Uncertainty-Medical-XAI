import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path

import config as cfg
from dataset import get_dataloaders
from models import get_model

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels, _ in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main(args):
    cfg.set_seed(cfg.SEED)
    
    print(f"Loading data...")
    train_loader, val_loader, _, train_ds = get_dataloaders(cfg.BATCH_SIZE)
    
    print(f"Building model: {args.model}")
    model = get_model(args.model)
    
    class_weights = train_ds.class_weights().to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    save_path = cfg.CKPT_DIR / f"{args.model}_best.pth"
    
    print(f"Starting training on {cfg.DEVICE}...")
    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, cfg.DEVICE)
        
        print(f"Epoch [{epoch}/{cfg.NUM_EPOCHS}] - "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
              
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best model to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline models.")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet18", "resnet50", "densenet121", "mc_resnet50"],
                        help="Model architecture to train.")
    args = parser.parse_args()
    main(args)
