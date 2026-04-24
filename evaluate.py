import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm

import config as cfg
from dataset import get_dataloaders
from models import get_model
from uncertainty import evaluate_metrics, calculate_entropy, plot_reliability_diagram

def get_predictions_deterministic(model, dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            
            all_probs.append(probs)
            all_labels.extend(labels.numpy())
            
    return np.vstack(all_probs), np.array(all_labels)

def get_predictions_mc_dropout(model, dataloader, device, mc_samples=30):
    model.eval()
    if hasattr(model, 'enable_dropout'):
        model.enable_dropout()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Evaluating MC Dropout"):
            images = images.to(device)
            
            mc_outputs = []
            for _ in range(mc_samples):
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                mc_outputs.append(probs.unsqueeze(0))
                
            mc_outputs = torch.cat(mc_outputs, dim=0)
            mean_probs = mc_outputs.mean(dim=0).cpu().numpy()
            
            all_probs.append(mean_probs)
            all_labels.extend(labels.numpy())
            
    return np.vstack(all_probs), np.array(all_labels)

def main(args):
    cfg.set_seed(cfg.SEED)
    
    _, _, test_loader, _ = get_dataloaders(cfg.BATCH_SIZE)
    
    print(f"Building model: {args.model}")
    model = get_model(args.model)
    model.load_state_dict(torch.load(args.checkpoint, map_location=cfg.DEVICE))
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    if args.model == "mc_resnet50":
        probs, labels = get_predictions_mc_dropout(model, test_loader, cfg.DEVICE, args.mc_samples)
        entropies = calculate_entropy(probs)
        print(f"Mean Predictive Entropy: {entropies.mean():.4f}")
    else:
        probs, labels = get_predictions_deterministic(model, test_loader, cfg.DEVICE)
        
    metrics = evaluate_metrics(probs, labels)
    
    print("\n--- Evaluation Results ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    # Save tables
    metrics_df = pd.DataFrame([metrics])
    metrics_csv_path = cfg.TABLES_DIR / f"{args.model}_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Saved metrics to {metrics_csv_path}")
    
    # Reliability diagram
    fig_path = cfg.FIGURES_DIR / f"{args.model}_reliability.png"
    plot_reliability_diagram(probs, labels, save_path=fig_path)
    print(f"Saved reliability diagram to {fig_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["resnet18", "resnet50", "densenet121", "mc_resnet50"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model weights")
    parser.add_argument("--mc_samples", type=int, default=cfg.MC_SAMPLES, help="Number of MC Dropout passes")
    args = parser.parse_args()
    main(args)
