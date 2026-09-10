"""
train_braingaze_model_v3.py
============================
Training script for BrainGaze-Diffusion v3.

Integrates insights from "The Modality Imbalance Report" into the training
loop.  All changes are additive to v2; original v1/v2 files are untouched.

Key changes vs. v2:
────────────────────
[1]  Adaptive Learning Rate (CLS-based):
     Uses the Conditional Learning Speed ratio (CLS_eeg / CLS_dec) to detect
     when the EEG branch is falling behind. When the ratio drops below a
     threshold, the EEG encoder's learning rate is boosted by 1.5×.
     This implements Solution 1 from the report (Balanced Multi-modal Learning).

[2]  Visual Noise Schedule:
     The model's VisualNoiseSchedule is updated each joint epoch, decaying
     from σ=0.3 to 0 over 25 epochs. This implements PER-style dominant
     modality regularization (Solution 2 in the report).

[3]  Gradient-aligned combined loss:
     Inspired by MMPareto (Solution 3): when the main saliency loss gradient
     and the EEG auxiliary loss gradient conflict (negative cosine similarity),
     we project the conflicting component out so gradients are never counter-
     productive.  This prevents the decoder from being pulled in contradictory
     directions by the visual vs. EEG objectives.

[4]  Three-phase curriculum:
     Phase 1 (epochs 1-10):   EEG-only warmup (same as v2)
     Phase 2 (epochs 11-30):  Joint training with visual noise + high aux weight
     Phase 3 (epochs 31-50):  Fine-tuning: noise decayed, lower aux weight,
                              discrimination loss still active

[5]  All outputs go to *_v3 directories.
"""

import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from src.braingaze_diffusion_model_v3 import BrainGazeDiffusionModel
from src.eeg_saliency_pipeline        import EEGSaliencyDataset


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════
EEG_DIR    = os.path.join(PROJECT_ROOT, "BGD_Dataset", "EEG")
STIM_ROOT  = os.path.join(PROJECT_ROOT, "BGD_Dataset", "images")
MAPS_ROOT  = os.path.join(PROJECT_ROOT, "BGD_Dataset", "maps")

MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "outputs", "models",
                               "braingaze_diffusion_model_v3.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS         = 50
WARMUP_EPOCHS  = 10
BATCH_SIZE     = 8

# Loss weights
LAMBDA_VICREG  = 3.0
LAMBDA_DISC    = 2.0
LAMBDA_AUX     = 0.5      # Phase 2 initial value (decays in Phase 3)

# CLS adaptive LR
CLS_RATIO_THRESHOLD = 0.3     # if EEG gradient ratio drops below this, boost LR
CLS_LR_BOOST        = 1.5     # multiplicative boost factor
CLS_CHECK_EVERY      = 3      # check every N epochs to avoid over-reacting


# ═══════════════════════════════════════════════════════════════════════════════
#  Loss Functions
# ═══════════════════════════════════════════════════════════════════════════════

def kld_loss(pred, target):
    eps = 1e-7
    p = torch.softmax(pred.view(pred.size(0), -1), dim=1)
    t = target.view(target.size(0), -1)
    return torch.sum(t * (torch.log(t + eps) - torch.log(p + eps)), dim=1).mean()


def cc_loss(pred, target):
    p = pred.view(pred.size(0), -1)
    t = target.view(target.size(0), -1)
    p = (p - p.mean(1, keepdim=True)) / (p.std(1, keepdim=True) + 1e-7)
    t = (t - t.mean(1, keepdim=True)) / (t.std(1, keepdim=True) + 1e-7)
    return -(p * t).mean(dim=1).mean()


def saliency_loss(pred, target):
    return kld_loss(pred, target) + 0.5 * cc_loss(pred, target)


def vicreg_variance_loss(eeg_feat):
    if eeg_feat.size(0) <= 1:
        return torch.tensor(0.0, device=eeg_feat.device)
    std = torch.std(eeg_feat, dim=0, unbiased=False)
    return torch.mean(torch.relu(1.0 - std))


def eeg_discrimination_loss(pred_real, pred_shuffled):
    """Penalises the model when different-EEG predictions are too similar."""
    p = pred_real.view(pred_real.size(0), -1)
    q = pred_shuffled.view(pred_shuffled.size(0), -1)
    p_n = (p - p.mean(1, keepdim=True)) / (p.std(1, keepdim=True) + 1e-7)
    q_n = (q - q.mean(1, keepdim=True)) / (q.std(1, keepdim=True) + 1e-7)
    per_sample_cc = (p_n * q_n).mean(dim=1)
    margin = 0.85
    return torch.relu(per_sample_cc - margin).mean()


# ═══════════════════════════════════════════════════════════════════════════════
#  Gradient Alignment  (MMPareto-inspired)
# ═══════════════════════════════════════════════════════════════════════════════

def align_gradients(model, loss_main, loss_aux):
    """
    If the main and auxiliary gradients conflict (negative cosine similarity),
    project out the conflicting component from the auxiliary gradient.

    This prevents the shared decoder parameters from being pulled in
    contradictory directions by the visual (main) and EEG (aux) objectives.

    Inspired by MMPareto: finds gradient direction common to both objectives.
    """
    # Get main gradient
    grad_main = torch.autograd.grad(loss_main, model.parameters(),
                                     retain_graph=True, allow_unused=True)
    # Get aux gradient
    grad_aux = torch.autograd.grad(loss_aux, model.parameters(),
                                    retain_graph=True, allow_unused=True)

    # Flatten to single vectors
    g_m = torch.cat([g.reshape(-1) for g in grad_main if g is not None])
    g_a = torch.cat([g.reshape(-1) for g in grad_aux  if g is not None])

    # Cosine similarity
    cos_sim = torch.dot(g_m, g_a) / (g_m.norm() * g_a.norm() + 1e-8)

    if cos_sim < 0:
        # Conflict:  project conflicting component out of g_a
        # g_a_aligned = g_a - (g_a · g_m / |g_m|²) · g_m
        proj = torch.dot(g_a, g_m) / (g_m.norm() ** 2 + 1e-8)
        g_a_aligned = g_a - proj * g_m

        # Apply aligned gradient manually
        offset = 0
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                n = p.numel()
                p.grad.data.add_(g_a_aligned[offset:offset + n].reshape(p.shape))
                offset += n
        return cos_sim.item(), True   # conflict detected, aligned
    else:
        # No conflict: just let both losses backprop normally
        loss_aux.backward(retain_graph=False)
        return cos_sim.item(), False


# ═══════════════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def build_optimizer(model, eeg_lr, dec_lr):
    eeg_params = (list(model.eeg_encoder.parameters())
                  + list(model.subject_embed.parameters()))
    dec_params = [p for name, p in model.named_parameters()
                  if 'eeg_encoder' not in name and 'subject_embed' not in name
                  and p.requires_grad]
    return optim.AdamW(
        [{'params': eeg_params, 'lr': eeg_lr},
         {'params': dec_params, 'lr': dec_lr}],
        weight_decay=1e-4
    )


def save_progression_image(model, track_data, epoch, out_dir, device):
    model.eval()
    eeg     = track_data['eeg'].unsqueeze(0).to(device)
    image   = track_data['image'].unsqueeze(0).to(device)
    subj    = torch.tensor([track_data['subject_id']]).to(device)

    with torch.no_grad():
        pred_joint = model(eeg, image, subj,
                           zero_image=False, apply_visual_noise=False)
        pred_eeg   = model(eeg, image, subj,
                           zero_image=True,  apply_visual_noise=False)

    def to_map(t):
        m = t.squeeze().cpu().numpy()
        return (m - m.min()) / (m.max() - m.min() + 1e-8)

    orig = track_data['image'].permute(1, 2, 0).numpy()
    orig = np.clip(orig * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax in axes:
        ax.axis('off')

    axes[0].imshow(orig)
    axes[0].set_title("Original Stimulus", fontsize=12)
    axes[1].imshow(orig); axes[1].imshow(to_map(pred_joint), cmap='jet', alpha=0.55)
    axes[1].set_title(f"Joint Prediction  (Epoch {epoch})", fontsize=12)
    axes[2].imshow(orig); axes[2].imshow(to_map(pred_eeg), cmap='jet', alpha=0.55)
    axes[2].set_title(f"EEG-Only Prediction  (Epoch {epoch})", fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"epoch_{epoch}.png"),
                bbox_inches='tight', dpi=130)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

def train():
    # Directories
    CHECKPOINT_DIR  = os.path.join(PROJECT_ROOT, "outputs", "models", "checkpoints_v3")
    PROGRESSION_DIR = os.path.join(PROJECT_ROOT, "outputs", "epoch_progression_v3")
    LOG_PATH        = os.path.join(PROJECT_ROOT, "outputs", "logs", "train_log_v3.csv")
    for d in [CHECKPOINT_DIR, PROGRESSION_DIR,
              os.path.dirname(MODEL_SAVE_PATH), os.path.dirname(LOG_PATH)]:
        os.makedirs(d, exist_ok=True)

    # Dataset
    print("Loading BrainGaze-Diffusion v3 dataset...")
    dataset    = EEGSaliencyDataset(EEG_DIR, STIM_ROOT, MAPS_ROOT, split="training")
    train_size = int(0.85 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Model
    model = BrainGazeDiffusionModel(
        noise_sigma=0.3, noise_decay_epochs=25
    ).to(DEVICE)

    track_sample = val_ds[0]

    # Initial optimizer
    optimizer = build_optimizer(model, eeg_lr=2e-3, dec_lr=5e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val = float('inf')
    log_lines = ["epoch,phase,train_loss,val_loss,disc,aux,vicreg,"
                 "eeg_gnorm,dec_gnorm,cls_ratio,cos_sim,vis_sigma,lr"]

    print(f"\n{'═'*70}")
    print(f"  BrainGaze v3 Training")
    print(f"  Epochs: {EPOCHS} | Warmup: {WARMUP_EPOCHS} | Device: {DEVICE}")
    print(f"  λ_VICReg={LAMBDA_VICREG} | λ_Disc={LAMBDA_DISC} | λ_Aux={LAMBDA_AUX}")
    print(f"{'═'*70}\n")

    for epoch in range(EPOCHS):
        # ── Phase determination ──────────────────────────────────────────────
        is_warmup = (epoch < WARMUP_EPOCHS)

        if epoch == WARMUP_EPOCHS:
            print(f"\n{'═'*70}")
            print("  Phase 2 → Joint Training.  Resetting optimizer.")
            print(f"{'═'*70}")
            optimizer = build_optimizer(model, eeg_lr=1e-3, dec_lr=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=EPOCHS - WARMUP_EPOCHS
            )

        # Phase 3 starts at epoch 30: reduce aux weight, noise is near zero
        phase = "Warmup" if is_warmup else ("Joint" if epoch < 30 else "Refine")
        aux_weight = LAMBDA_AUX if epoch < 30 else LAMBDA_AUX * 0.3

        # Update visual noise schedule
        if not is_warmup:
            model.visual_noise.set_epoch(epoch - WARMUP_EPOCHS)

        vis_sigma = model.visual_noise.current_sigma.item()
        print(f"\n── Epoch {epoch+1:>2}/{EPOCHS}  [{phase}]  σ_noise={vis_sigma:.3f} ─")

        # ════════════════════════════════════════════════════════════════════
        #  TRAINING
        # ════════════════════════════════════════════════════════════════════
        model.train()
        run = {k: 0.0 for k in ['sal', 'disc', 'aux', 'var', 'total',
                                 'cos_sim', 'conflicts']}

        loop = tqdm(train_loader, desc="[Train]", leave=False)
        for batch in loop:
            eeg         = batch['eeg'].to(DEVICE)
            image       = batch['image'].to(DEVICE)
            target      = batch['saliency'].to(DEVICE)
            subject_ids = batch['subject_id'].to(DEVICE)

            optimizer.zero_grad()

            if is_warmup:
                # Phase 1: EEG-only warmup
                out  = model(eeg, image, subject_ids,
                             zero_image=True, apply_visual_noise=False)
                loss = saliency_loss(out, target)

                eeg_feat = model.get_eeg_features(eeg, subject_ids)
                v_loss   = vicreg_variance_loss(eeg_feat)
                loss     = loss + LAMBDA_VICREG * v_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                run['sal']   += loss.item()
                run['total'] += loss.item()
                loop.set_postfix(sal=f"{loss.item():.3f}")

            else:
                # Phase 2/3: Joint training

                # ── Pass A: main (real EEG + real image) ─────────────────
                out_real = model(eeg, image, subject_ids,
                                 zero_image=False, apply_visual_noise=True)
                loss_sal = saliency_loss(out_real, target)

                # ── Pass B: discrimination (shuffled EEG, same images) ───
                perm = torch.randperm(eeg.size(0), device=DEVICE)
                with torch.no_grad():
                    out_shuf = model(eeg[perm], image, subject_ids[perm],
                                     zero_image=False, apply_visual_noise=False)
                loss_disc = eeg_discrimination_loss(out_real, out_shuf)

                # ── Pass C: EEG-only auxiliary ───────────────────────────
                out_eeg = model(eeg, image, subject_ids,
                                zero_image=True, apply_visual_noise=False)
                loss_aux = saliency_loss(out_eeg, target)

                # ── VICReg ───────────────────────────────────────────────
                eeg_feat = model.get_eeg_features(eeg, subject_ids)
                loss_var = vicreg_variance_loss(eeg_feat)

                # ── Combined loss with gradient alignment ────────────────
                #    Backprop main loss first
                loss_main = loss_sal + LAMBDA_DISC * loss_disc + LAMBDA_VICREG * loss_var
                loss_main.backward(retain_graph=True)

                #    Align aux gradient with main if they conflict
                loss_aux_weighted = aux_weight * loss_aux
                cos_sim, conflict = align_gradients(
                    model, loss_main, loss_aux_weighted
                )

                total = loss_main.item() + loss_aux_weighted.item()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                run['sal']       += loss_sal.item()
                run['disc']      += loss_disc.item()
                run['aux']       += loss_aux.item()
                run['var']       += loss_var.item()
                run['total']     += total
                run['cos_sim']   += cos_sim
                run['conflicts'] += int(conflict)

                loop.set_postfix(
                    sal=f"{loss_sal.item():.3f}",
                    disc=f"{loss_disc.item():.3f}",
                    aux=f"{loss_aux.item():.3f}",
                    cos=f"{cos_sim:.2f}",
                )

        # ── CLS-based adaptive LR  ──────────────────────────────────────────
        eeg_gnorm, dec_gnorm = model.get_gradient_norms()
        cls_ratio = eeg_gnorm / (dec_gnorm + 1e-8)

        if (not is_warmup
            and (epoch - WARMUP_EPOCHS) % CLS_CHECK_EVERY == 0
            and cls_ratio < CLS_RATIO_THRESHOLD):
            old_lr = optimizer.param_groups[0]['lr']
            new_lr = old_lr * CLS_LR_BOOST
            optimizer.param_groups[0]['lr'] = new_lr
            print(f"  ⚡ CLS ratio {cls_ratio:.3f} < {CLS_RATIO_THRESHOLD} "
                  f"→ EEG LR boosted {old_lr:.6f} → {new_lr:.6f}")

        scheduler.step()

        # ════════════════════════════════════════════════════════════════════
        #  VALIDATION
        # ════════════════════════════════════════════════════════════════════
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                eeg   = batch['eeg'].to(DEVICE)
                image = batch['image'].to(DEVICE)
                tgt   = batch['saliency'].to(DEVICE)
                subs  = batch['subject_id'].to(DEVICE)
                out   = model(eeg, image, subs,
                              zero_image=is_warmup, apply_visual_noise=False)
                val_loss += saliency_loss(out, tgt).item()

        n   = len(train_loader)
        nv  = len(val_loader)
        avg = {k: v / n for k, v in run.items()}
        avg_val = val_loss / nv
        cur_lr  = scheduler.get_last_lr()[0]

        print(f"  Train: {avg['total']:.4f} | Val: {avg_val:.4f} | "
              f"Disc: {avg['disc']:.4f} | Aux: {avg['aux']:.4f} | "
              f"CLS: {cls_ratio:.3f} | cos: {avg['cos_sim']:.3f} | "
              f"σ: {vis_sigma:.3f} | LR: {cur_lr:.6f}")

        log_lines.append(
            f"{epoch+1},{phase},{avg['total']:.4f},{avg_val:.4f},"
            f"{avg['disc']:.4f},{avg['aux']:.4f},{avg['var']:.4f},"
            f"{eeg_gnorm:.5f},{dec_gnorm:.5f},{cls_ratio:.4f},"
            f"{avg['cos_sim']:.4f},{vis_sigma:.4f},{cur_lr:.6f}"
        )

        # ── Checkpoints & progression ────────────────────────────────────────
        torch.save(model.state_dict(),
                   os.path.join(CHECKPOINT_DIR, f"braingaze_v3_epoch_{epoch+1}.pth"))
        save_progression_image(model, track_sample, epoch+1, PROGRESSION_DIR, DEVICE)

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ──▶  Best model saved  (val={avg_val:.4f})")

    # Write log
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

    print(f"\n{'═'*60}")
    print(f"  BrainGaze v3 Training Complete!")
    print(f"  Best val loss: {best_val:.4f}")
    print(f"  Model: {MODEL_SAVE_PATH}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    train()
