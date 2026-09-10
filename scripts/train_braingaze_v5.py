"""
train_braingaze_v5.py
=====================
Production-grade 50-epoch training script for BrainGaze:
Cognitive-Visual Modular Routing (CVMR) Architecture.

Key Highlights:
- Modality: Multimodal (32-ch EEG waveforms + ResNet-18 visual feature decoders)
- Saliency Losses: CC (Correlation Coefficient) + KLD (Kullback-Leibler Divergence)
- Regularization:
    1. Parseval Orthogonality Loss (L_ortho): enforces spatial basis decorrelation.
    2. Cognitive Diversity Loss (L_div): prevents router collapse and maintains entropy.
- Checkpointing:
    - Periodic checkpoints in outputs/models/checkpoints_v5/
    - Best validation checkpoint: outputs/models/braingaze_v5_best.pth
    - Final model weights: outputs/models/braingaze_v5.pth
- Diagnostics & Visual Tracking:
    - Real-time tqdm progress bars
    - Epoch-by-epoch visual basis decomposition in outputs/epoch_progression_v5/
    - Loss and metric logging in outputs/logs/train_log_v5.csv
    - Publication-grade training curve plots in outputs/figures/train_v5_curves.png
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# Set project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from src.braingaze_v5 import BrainGaze_v5_CVMR, loss_orthogonality, loss_routing_diversity
from src.eeg_saliency_pipeline import EEGSaliencyDataset


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics & Loss Formulations
# ═══════════════════════════════════════════════════════════════════════════════

def cc_metric(pred, target):
    """
    Computes Pearson Correlation Coefficient (CC) across spatial dimensions.
    Range: [-1.0, 1.0], higher is better.
    """
    p = pred.view(pred.size(0), -1)
    t = target.view(target.size(0), -1)
    p_n = (p - p.mean(1, keepdim=True)) / (p.std(1, keepdim=True) + 1e-7)
    t_n = (t - t.mean(1, keepdim=True)) / (t.std(1, keepdim=True) + 1e-7)
    return (p_n * t_n).mean(dim=1).mean()


def kld_loss(pred, target):
    """
    Kullback-Leibler Divergence (KLD) between predicted and ground-truth distributions.
    """
    eps = 1e-7
    p = pred.view(pred.size(0), -1)
    p = p / (p.sum(dim=1, keepdim=True) + eps)
    t = target.view(target.size(0), -1)
    t = t / (t.sum(dim=1, keepdim=True) + eps)
    return torch.sum(t * (torch.log(t + eps) - torch.log(p + eps)), dim=1).mean()


def compute_entropy(weights):
    """Computes Shannon Entropy H(alpha) in nats."""
    eps = 1e-8
    return -torch.sum(weights * torch.log(weights + eps), dim=-1).mean()


def compute_gram_similarity(basis_maps):
    """Computes mean off-diagonal Gram cosine similarity across basis maps."""
    B, K, H, W = basis_maps.shape
    flat = basis_maps.view(B, K, -1)
    flat_norm = flat / (flat.norm(dim=-1, keepdim=True) + 1e-7)
    gram = torch.bmm(flat_norm, flat_norm.transpose(1, 2))  # (B, K, K)
    mask = ~torch.eye(K, dtype=torch.bool, device=basis_maps.device).unsqueeze(0)
    off_diag = gram[mask.expand_as(gram)].view(B, -1)
    return off_diag.abs().mean()


# ═══════════════════════════════════════════════════════════════════════════════
# Progression Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def save_epoch_visualization(model, sample, epoch, output_dir, device):
    """
    Renders visual basis decomposition and predicted vs. ground truth maps.
    Saved to output_dir/epoch_{epoch:02d}.png.
    """
    model.eval()
    with torch.no_grad():
        img = sample['image'].unsqueeze(0).to(device)
        eeg = sample['eeg'].unsqueeze(0).to(device)
        gt = sample['saliency'].unsqueeze(0).to(device)
        subj = torch.tensor([sample['subject_id']]).to(device)

        pred, basis_maps, routing_weights, _, _ = model(eeg, img, subj)

        cc_val = cc_metric(pred, gt).item()
        alpha = routing_weights.squeeze().cpu().numpy()

        # Un-normalize image
        img_np = sample['image'].permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_rgb = np.clip(img_np * std + mean, 0, 1)

        gt_map = gt.squeeze().cpu().numpy()
        gt_norm = (gt_map - gt_map.min()) / (gt_map.max() - gt_map.min() + 1e-8)

        pred_map = pred.squeeze().cpu().numpy()
        pred_norm = (pred_map - pred_map.min()) / (pred_map.max() - pred_map.min() + 1e-8)

        # Plot 3x4 grid: Top row: Stimulus, GT, Pred, Empty. Bottom 2 rows: 8 Basis Maps
        fig = plt.figure(figsize=(16, 12))

        # 1. Stimulus
        ax1 = plt.subplot2grid((3, 4), (0, 0))
        ax1.imshow(img_rgb)
        ax1.set_title("Input Stimulus (COCO)", fontsize=11, fontweight='bold')
        ax1.axis('off')

        # 2. Ground Truth
        ax2 = plt.subplot2grid((3, 4), (0, 1))
        ax2.imshow(img_rgb)
        ax2.imshow(gt_norm, cmap='jet', alpha=0.55)
        ax2.set_title("Ground Truth Saliency", fontsize=11, fontweight='bold')
        ax2.axis('off')

        # 3. Model Prediction
        ax3 = plt.subplot2grid((3, 4), (0, 2))
        ax3.imshow(img_rgb)
        ax3.imshow(pred_norm, cmap='jet', alpha=0.55)
        ax3.set_title(f"BrainGaze CVMR (Epoch {epoch})\nCC: {cc_val:.4f}", fontsize=11, fontweight='bold', color='darkgreen')
        ax3.axis('off')

        # 4. Routing Weights Bar Chart
        ax4 = plt.subplot2grid((3, 4), (0, 3))
        bars = ax4.bar(range(1, 9), alpha, color='#1f77b4', edgecolor='black', alpha=0.85)
        ax4.set_title(f"Cognitive Routing Weights (α)\nEntropy: {compute_entropy(routing_weights).item():.3f} nats", fontsize=10, fontweight='bold')
        ax4.set_xlabel("Basis Index k")
        ax4.set_ylabel("Weight α_k")
        ax4.set_ylim(0, max(0.35, float(alpha.max()) + 0.05))
        ax4.grid(True, linestyle='--', alpha=0.3)
        for bar in bars:
            yval = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom', fontsize=8)

        # 8 Basis Maps
        basis_np = basis_maps.squeeze(0).cpu().numpy() # (8, 224, 224)
        for k in range(8):
            row = 1 + (k // 4)
            col = k % 4
            ax = plt.subplot2grid((3, 4), (row, col))
            b_map = basis_np[k]
            b_norm = (b_map - b_map.min()) / (b_map.max() - b_map.min() + 1e-8)
            ax.imshow(img_rgb)
            ax.imshow(b_norm, cmap='magma', alpha=0.6)
            ax.set_title(f"Basis M_{k+1} (α_{k+1} = {alpha[k]:.3f})", fontsize=10, fontweight='bold')
            ax.axis('off')

        plt.suptitle(f"BrainGaze v5 (CVMR) Visual Basis Progression — Epoch {epoch:02d}", fontsize=14, fontweight='bold', y=0.99)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"epoch_{epoch:02d}.png")
        plt.savefig(save_path, dpi=130, bbox_inches='tight')
        plt.close()


def plot_training_curves(history, output_path):
    """Plots training and validation loss and metric curves."""
    epochs = [h['epoch'] for h in history]
    if not epochs:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss Curves
    axes[0, 0].plot(epochs, [h['train_loss'] for h in history], 'b-o', label='Train Loss', linewidth=1.8, markersize=4)
    axes[0, 0].plot(epochs, [h['val_loss'] for h in history], 'r-s', label='Val Loss', linewidth=1.8, markersize=4)
    axes[0, 0].set_title("Total Training & Validation Loss", fontweight='bold')
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, linestyle='--', alpha=0.4)
    axes[0, 0].legend()

    # Correlation (CC) Curves
    axes[0, 1].plot(epochs, [h['train_cc'] for h in history], 'b-o', label='Train CC', linewidth=1.8, markersize=4)
    axes[0, 1].plot(epochs, [h['val_cc'] for h in history], 'g-^', label='Val CC', linewidth=2.0, markersize=5)
    axes[0, 1].axhline(0.85, color='gray', linestyle=':', label='Target Threshold (0.85)')
    axes[0, 1].set_title("Pearson Correlation Coefficient (CC)", fontweight='bold')
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("CC")
    axes[0, 1].set_ylim(-0.1, 1.0)
    axes[0, 1].grid(True, linestyle='--', alpha=0.4)
    axes[0, 1].legend()

    # Parseval Orthogonality Score
    axes[1, 0].plot(epochs, [h['val_ortho'] for h in history], 'm-d', label='Basis Gram Similarity (L_ortho)', linewidth=1.8, markersize=4)
    axes[1, 0].set_title("Spatial Basis Orthogonality (Off-Diag Similarity)", fontweight='bold')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Gram Similarity (Lower = More Orthogonal)")
    axes[1, 0].grid(True, linestyle='--', alpha=0.4)
    axes[1, 0].legend()

    # Cognitive Routing Variance / Entropy
    axes[1, 1].plot(epochs, [h['val_entropy'] for h in history], 'c-p', label='Routing Shannon Entropy H(α)', linewidth=1.8, markersize=4)
    axes[1, 1].axhline(np.log(8), color='red', linestyle='--', label=f'Max Capacity ln(8)={np.log(8):.3f}')
    axes[1, 1].set_title("Cognitive Routing Policy Entropy", fontweight='bold')
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Entropy (nats)")
    axes[1, 1].grid(True, linestyle='--', alpha=0.4)
    axes[1, 1].legend()

    plt.suptitle("BrainGaze Architecture v5 (CVMR) Training Dynamics", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Main Training Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train BrainGaze v5 (CVMR) across entire dataset.")
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam (default: 0.001)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--lambda_ortho", type=float, default=0.3, help="Parseval Orthogonality loss weight (default: 0.3)")
    parser.add_argument("--lambda_div", type=float, default=0.8, help="Cognitive Routing diversity loss weight (default: 0.8)")
    parser.add_argument("--num_basis", type=int, default=8, help="Number of spatial basis maps (default: 8)")
    parser.add_argument("--num_workers", type=int, default=0, help="Data loader workers (default: 0 for Windows compatibility)")
    parser.add_argument("--device", type=str, default="", help="Device: 'cuda', 'cpu', or '' (auto-detect)")
    parser.add_argument("--max_samples", type=int, default=0, help="Optional subset cap for fast testing (0 = full dataset)")
    parser.add_argument("--save_interval", type=int, default=5, help="Save periodic checkpoint every N epochs (default: 5)")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Determine Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("BRAINGAZE ARCHITECTURE v5: FULL DATASET 50-EPOCH TRAINING PIPELINE")
    print(f"Device: {device} | Total Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print("=" * 80)

    # Setup directories
    eeg_dir = os.path.join(PROJECT_ROOT, "BGD_Dataset", "EEG")
    stim_root = os.path.join(PROJECT_ROOT, "BGD_Dataset", "images")
    maps_root = os.path.join(PROJECT_ROOT, "BGD_Dataset", "maps")

    models_dir = os.path.join(PROJECT_ROOT, "outputs", "models")
    checkpoints_dir = os.path.join(models_dir, "checkpoints_v5")
    progression_dir = os.path.join(PROJECT_ROOT, "outputs", "epoch_progression_v5")
    logs_dir = os.path.join(PROJECT_ROOT, "outputs", "logs")
    figures_dir = os.path.join(PROJECT_ROOT, "outputs", "figures")

    for d in [models_dir, checkpoints_dir, progression_dir, logs_dir, figures_dir]:
        os.makedirs(d, exist_ok=True)

    log_csv_path = os.path.join(logs_dir, "train_log_v5.csv")
    final_model_path = os.path.join(models_dir, "braingaze_v5.pth")
    best_model_path = os.path.join(models_dir, "braingaze_v5_best.pth")
    curves_plot_path = os.path.join(figures_dir, "train_v5_curves.png")

    # 1. Load Dataset
    print("\n[Step 1/5] Loading Full BGD Dataset...")
    train_dataset = EEGSaliencyDataset(eeg_dir, stim_root, maps_root, split="training")
    val_dataset = EEGSaliencyDataset(eeg_dir, stim_root, maps_root, split="test")

    if args.max_samples > 0:
        train_dataset = Subset(train_dataset, range(min(args.max_samples, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(args.max_samples // 5, len(val_dataset))))
        print(f"--> Debug Subset active: {len(train_dataset)} train, {len(val_dataset)} val.")
    else:
        print(f"--> Full Dataset Loaded: {len(train_dataset):,} train samples, {len(val_dataset):,} validation samples.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )

    # Reference sample for visual progression
    tracking_sample = val_dataset[0]

    # 2. Instantiate Model
    print("\n[Step 2/5] Initializing BrainGaze v5 (CVMR) Architecture...")
    model = BrainGaze_v5_CVMR(num_basis=args.num_basis).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--> Total Parameters: {total_params:,} | Trainable: {trainable_params:,}")

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 1
    best_val_cc = -1.0
    history = []

    # Optional Resume
    if args.resume and os.path.exists(args.resume):
        print(f"--> Resuming checkpoint from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_cc = checkpoint.get('best_val_cc', -1.0)
        history = checkpoint.get('history', [])
        print(f"--> Resumed from epoch {start_epoch} (Best Val CC: {best_val_cc:.4f})")

    # Prepare CSV Log
    if not os.path.exists(log_csv_path) or start_epoch == 1:
        with open(log_csv_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,train_cc,train_ortho,val_loss,val_cc,val_kld,val_ortho,val_entropy,lr,time_sec\n")

    print("\n[Step 3/5] Starting 50-Epoch Training Loop...")
    start_total_time = time.time()

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start_time = time.time()
            model.train()

            train_losses = []
            train_ccs = []
            train_orthos = []

            pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]", leave=True)
            for batch in pbar:
                eeg = batch['eeg'].to(device)
                img = batch['image'].to(device)
                gt = batch['saliency'].to(device)
                subj = batch['subject_id'].to(device)

                optimizer.zero_grad()

                pred, basis_maps, routing_weights, _, _ = model(eeg, img, subj)

                # Primary Saliency Fidelity (CC + KLD)
                cc_val = cc_metric(pred, gt)
                loss_cc = -cc_val
                loss_kld_val = kld_loss(pred, gt)
                loss_sal = loss_cc + 0.5 * loss_kld_val

                # Regularization Terms
                loss_ortho = loss_orthogonality(basis_maps)
                loss_div = loss_routing_diversity(routing_weights)

                total_loss = loss_sal + args.lambda_ortho * loss_ortho + args.lambda_div * loss_div
                total_loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

                train_losses.append(total_loss.item())
                train_ccs.append(cc_val.item())
                train_orthos.append(loss_ortho.item())

                pbar.set_postfix({
                    "Loss": f"{total_loss.item():.4f}",
                    "CC": f"{cc_val.item():.4f}",
                    "Ortho": f"{loss_ortho.item():.4f}",
                    "Std(α)": f"{torch.std(routing_weights, dim=0).mean().item():.3f}"
                })

            scheduler.step()

            # Validation Phase
            model.eval()
            val_losses = []
            val_ccs = []
            val_klds = []
            val_orthos = []
            val_entropies = []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Val]  ", leave=False):
                    eeg = batch['eeg'].to(device)
                    img = batch['image'].to(device)
                    gt = batch['saliency'].to(device)
                    subj = batch['subject_id'].to(device)

                    pred, basis_maps, routing_weights, _, _ = model(eeg, img, subj)

                    cc_val = cc_metric(pred, gt)
                    kld_val = kld_loss(pred, gt)
                    loss_ortho = loss_orthogonality(basis_maps)
                    loss_div = loss_routing_diversity(routing_weights)

                    total_val_loss = (-cc_val + 0.5 * kld_val) + args.lambda_ortho * loss_ortho + args.lambda_div * loss_div

                    val_losses.append(total_val_loss.item())
                    val_ccs.append(cc_val.item())
                    val_klds.append(kld_val.item())
                    val_orthos.append(compute_gram_similarity(basis_maps).item())
                    val_entropies.append(compute_entropy(routing_weights).item())

            # Aggregate Epoch Metrics
            epoch_duration = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]['lr']

            epoch_train_loss = float(np.mean(train_losses))
            epoch_train_cc = float(np.mean(train_ccs))
            epoch_train_ortho = float(np.mean(train_orthos))

            epoch_val_loss = float(np.mean(val_losses))
            epoch_val_cc = float(np.mean(val_ccs))
            epoch_val_kld = float(np.mean(val_klds))
            epoch_val_ortho = float(np.mean(val_orthos))
            epoch_val_entropy = float(np.mean(val_entropies))

            history.append({
                "epoch": epoch,
                "train_loss": epoch_train_loss,
                "train_cc": epoch_train_cc,
                "val_loss": epoch_val_loss,
                "val_cc": epoch_val_cc,
                "val_kld": epoch_val_kld,
                "val_ortho": epoch_val_ortho,
                "val_entropy": epoch_val_entropy,
                "lr": current_lr
            })

            # Print Summary
            print(f"\n>>> Epoch {epoch:02d}/{args.epochs:02d} Summary [{epoch_duration:.1f}s]:")
            print(f"    Train Loss: {epoch_train_loss:.4f} | Train CC: {epoch_train_cc:.4f} | Ortho Loss: {epoch_train_ortho:.4f}")
            print(f"    Val Loss:   {epoch_val_loss:.4f} | Val CC:   {epoch_val_cc:.4f} | Val KLD:    {epoch_val_kld:.4f}")
            print(f"    Gram Ortho: {epoch_val_ortho:.4f} | Entropy:  {epoch_val_entropy:.3f} | LR: {current_lr:.6f}")

            # Append to CSV log
            with open(log_csv_path, "a", encoding="utf-8") as f:
                f.write(f"{epoch},{epoch_train_loss:.5f},{epoch_train_cc:.5f},{epoch_train_ortho:.5f},"
                        f"{epoch_val_loss:.5f},{epoch_val_cc:.5f},{epoch_val_kld:.5f},{epoch_val_ortho:.5f},"
                        f"{epoch_val_entropy:.5f},{current_lr:.8f},{epoch_duration:.1f}\n")

            # Save Visual Basis Progression
            save_epoch_visualization(model, tracking_sample, epoch, progression_dir, device)

            # Update Plots
            plot_training_curves(history, curves_plot_path)

            # Save Best Model Checkpoint
            if epoch_val_cc > best_val_cc:
                best_val_cc = epoch_val_cc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_cc': epoch_val_cc,
                    'best_val_cc': best_val_cc,
                    'history': history,
                    'args': vars(args)
                }, best_model_path)
                print(f"    [★] New best validation CC ({best_val_cc:.4f})! Saved to {best_model_path}")

            # Save Periodic Checkpoint
            if epoch % args.save_interval == 0 or epoch == args.epochs:
                ckpt_path = os.path.join(checkpoints_dir, f"braingaze_v5_epoch_{epoch:02d}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_cc': epoch_val_cc,
                    'history': history,
                    'args': vars(args)
                }, ckpt_path)
                print(f"    Checkpoint saved to: {ckpt_path}")

    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user (KeyboardInterrupt). Saving emergency checkpoint...")
        emergency_path = os.path.join(models_dir, "braingaze_v5_interrupted.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'args': vars(args)
        }, emergency_path)
        print(f"    Emergency state saved to: {emergency_path}")

    # Final Save
    total_elapsed = time.time() - start_total_time
    torch.save(model.state_dict(), final_model_path)
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETED IN {total_elapsed / 3600:.2f} HOURS.")
    print(f"Final Weights Saved to:    {final_model_path}")
    print(f"Best Validation Checkpoint: {best_model_path} (Best Val CC: {best_val_cc:.4f})")
    print(f"Visual Progressions:        {progression_dir}")
    print(f"Training Curves Plot:       {curves_plot_path}")
    print(f"Complete Log File:          {log_csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
