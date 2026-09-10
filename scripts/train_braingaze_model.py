import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from src.braingaze_diffusion_model import BrainGazeDiffusionModel
from src.eeg_saliency_pipeline import EEGSaliencyDataset

# Configuration
EEG_DIR = os.path.join(PROJECT_ROOT, "BGD_Dataset", "EEG")
STIM_ROOT = os.path.join(PROJECT_ROOT, "BGD_Dataset", "images")
MAPS_ROOT = os.path.join(PROJECT_ROOT, "BGD_Dataset", "maps")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", "braingaze_diffusion_model_v1.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 30           # 30 epochs is typically sufficient for modulation to converge
BATCH_SIZE = 8
LR = 1e-4

def kld_loss(pred, target):
    eps = 1e-7
    p = torch.softmax(pred.view(pred.size(0), -1), dim=1)
    t = target.view(target.size(0), -1)
    return torch.sum(t * (torch.log(t + eps) - torch.log(p + eps)), dim=1).mean()

def cc_loss(pred, target):
    p = pred.view(pred.size(0), -1)
    t = target.view(target.size(0), -1)
    p_mu, p_std = p.mean(dim=1, keepdim=True), p.std(dim=1, keepdim=True)
    t_mu, t_std = t.mean(dim=1, keepdim=True), t.std(dim=1, keepdim=True)
    p = (p - p_mu) / (p_std + 1e-7)
    t = (t - t_mu) / (t_std + 1e-7)
    return -(p * t).mean(dim=1).mean()

def train():
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "outputs", "models", "checkpoints")
    PROGRESSION_DIR = os.path.join(PROJECT_ROOT, "outputs", "epoch_progression")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PROGRESSION_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    print("Loading BrainGaze-Diffusion matched dataset...")
    dataset = EEGSaliencyDataset(EEG_DIR, STIM_ROOT, MAPS_ROOT, split="training")
    
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    print(f"Training set: {len(train_ds)} samples | Validation set: {len(val_ds)} samples")
    
    # Initialize BrainGaze hybrid model
    model = BrainGazeDiffusionModel().to(DEVICE)
    
    # Choose a fixed validation sample to track visual progression epoch-by-epoch
    track_sample = val_ds[0]
    track_eeg = track_sample['eeg'].unsqueeze(0).to(DEVICE)
    track_image = track_sample['image'].unsqueeze(0).to(DEVICE)
    track_subject = torch.tensor([track_sample['subject_id']]).to(DEVICE)
    track_coco_id = track_sample['coco_id']
    
    # Separated Modality Learning Rates to prevent visual prior dominance
    eeg_params = list(model.eeg_encoder.parameters()) + list(model.subject_embed.parameters())
    decoder_film_params = [
        p for name, p in model.named_parameters()
        if "eeg_encoder" not in name and "subject_embed" not in name and p.requires_grad
    ]
    
    optimizer = optim.AdamW([
        {'params': eeg_params, 'lr': 1e-3},
        {'params': decoder_film_params, 'lr': 1e-4}
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_loss = float('inf')
    
    print(f"Starting BrainGaze-Diffusion Training on {DEVICE}...")
    for epoch in range(EPOCHS):
        model.train()
        train_l = 0.0
        
        # Phase check: Increased Warmup to 10 epochs
        is_warmup = (epoch < 10)
        phase_desc = "Phase 1: EEG Warmup (Images Muted)" if is_warmup else "Phase 2: Joint Training (Images Active)"
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} | {phase_desc} ---")
        
        loop = tqdm(train_loader, desc="[Train]", leave=False)
        for batch in loop:
            eeg = batch['eeg'].to(DEVICE)
            image = batch['image'].to(DEVICE)
            target = batch['saliency'].to(DEVICE)
            subject_ids = batch['subject_id'].to(DEVICE)
            
            if is_warmup:
                image = torch.zeros_like(image)
                
            optimizer.zero_grad()
            out = model(eeg, image, subject_ids, zero_image=is_warmup)
            
            # Compute VICReg Variance Regularization safely (skip if batch size is 1)
            eeg_feat = model.eeg_encoder(eeg)
            if eeg_feat.size(0) > 1:
                std_eeg = torch.std(eeg_feat, dim=0, unbiased=False)
                var_loss = torch.mean(torch.relu(1.0 - std_eeg))
            else:
                var_loss = 0.0
                std_eeg = torch.zeros(1).to(DEVICE)
            
            # Loss = KLD (distribution metric) + 0.5 * CC (correlation metric) + 2.0 * EEG Variance Regularization
            loss = kld_loss(out, target) + 0.5 * cc_loss(out, target) + 2.0 * var_loss
            loss.backward()
            optimizer.step()
            train_l += loss.item()
            loop.set_postfix(loss=loss.item(), eeg_std=torch.mean(std_eeg).item())
            
        scheduler.step()
        
        # Validation Phase
        model.eval()
        val_l = 0.0
        with torch.no_grad():
            for batch in val_loader:
                eeg = batch['eeg'].to(DEVICE)
                image = batch['image'].to(DEVICE)
                target = batch['saliency'].to(DEVICE)
                subject_ids = batch['subject_id'].to(DEVICE)
                
                if is_warmup:
                    image = torch.zeros_like(image)
                    
                out = model(eeg, image, subject_ids, zero_image=is_warmup)
                
                # Compute EEG Variance safely in validation loop
                eeg_feat = model.eeg_encoder(eeg)
                if eeg_feat.size(0) > 1:
                    std_eeg = torch.std(eeg_feat, dim=0, unbiased=False)
                    var_loss = torch.mean(torch.relu(1.0 - std_eeg))
                else:
                    var_loss = 0.0
                
                loss = kld_loss(out, target) + 0.5 * cc_loss(out, target) + 2.0 * var_loss
                val_l += loss.item()
                
        avg_train = train_l / len(train_loader)
        avg_val = val_l / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 1. Save checkpoint for the current epoch (regardless of loss) so user can show progression
        epoch_checkpoint_path = os.path.join(PROJECT_ROOT, "outputs", "models", "checkpoints", f"braingaze_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_checkpoint_path)
        
        # 2. Run inference on chosen sample to save epoch progression image
        model.eval()
        with torch.no_grad():
            track_out = model(track_eeg, track_image, track_subject, zero_image=is_warmup)
        pred_map = track_out.squeeze().cpu().numpy()
        pred_map = (pred_map - pred_map.min()) / (pred_map.max() - pred_map.min() + 1e-8)
        
        # Plot stimulus and predicted overlay
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        orig_img = track_sample['image'].permute(1, 2, 0).cpu().numpy()
        # Un-normalize image
        orig_img = orig_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        orig_img = np.clip(orig_img, 0, 1)
        
        ax[0].imshow(orig_img)
        ax[0].set_title("Original Stimulus")
        ax[0].axis('off')
        
        ax[1].imshow(orig_img)
        ax[1].imshow(pred_map, cmap='jet', alpha=0.5)
        ax[1].set_title(f"BrainGaze Prediction - Epoch {epoch+1}")
        ax[1].axis('off')
        
        prog_path = os.path.join(PROJECT_ROOT, "outputs", "epoch_progression", f"epoch_{epoch+1}.png")
        plt.savefig(prog_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        # 3. Save best checkpoint based on validation loss
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  --> Saved SOTA BrainGaze Model Checkpoint.")
            
    print("\nBrainGaze Hybrid Model Training Complete!")

if __name__ == "__main__":
    train()
