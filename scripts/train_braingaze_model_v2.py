"""
train_braingaze_model_v2.py
===========================
Revised training script for BrainGaze-Diffusion v2.

Key fixes over v1 (train_braingaze_model.py):
─────────────────────────────────────────────
[FIX 1] EEG Discrimination Loss
   The single most important fix. On every joint-training batch we shuffle the
   EEG signals within the batch (same images, different brains) and penalise
   the model if it produces similar saliency maps for the two inputs.
   This directly forces the FiLM layers to make EEG content affect the output.

[FIX 2] EEG-Only Auxiliary Loss
   An extra forward pass with zero_image=True is computed every batch during
   joint training and contributes to the loss. This forces the EEG encoder +
   FiLM path to independently predict plausible saliency rather than becoming
   a dead branch that the decoder ignores.

[FIX 3] Modality dropout removed from training loop
   v1 had `if (self.training and torch.rand(1).item() < 0.5): zero(feat*)` 
   inside model.forward(). This has been eliminated. The training script now
   has explicit, deterministic control over when visual features are zeroed
   via the zero_image and apply_spatial_dropout arguments.

[FIX 4] Optimizer reset at phase boundary
   When training transitions from Phase 1 (warmup) to Phase 2 (joint), a
   fresh AdamW optimizer is created with new LRs and momentum state cleared.
   This prevents stale momentum from the warmup phase from corrupting joint
   training gradients.

[FIX 5] Extended training and increased VICReg weight
   Epochs: 30 → 50  (10 warmup + 40 joint)
   VICReg λ: 2.0 → 3.0  (stronger push to keep EEG embeddings diverse)

[FIX 6] Gradient norm monitoring
   EEG encoder gradient norms are logged per epoch. Near-zero norms indicate
   the bypass is still active; growing norms indicate the discrimination and
   auxiliary losses are working.

[FIX 7] Epoch progression shows both joint and EEG-only predictions
   The saved PNG now shows three panels: original, joint prediction, EEG-only
   prediction — making the EEG contribution visually traceable across epochs.

[FIX 8] All outputs saved to *_v2 directories / filenames
   The original v1 model, checkpoints, and logs are never touched.
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

from src.braingaze_diffusion_model_v2 import BrainGazeDiffusionModel
from src.eeg_saliency_pipeline        import EEGSaliencyDataset

# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════
EEG_DIR    = os.path.join(PROJECT_ROOT, "BGD_Dataset", "EEG")
STIM_ROOT  = os.path.join(PROJECT_ROOT, "BGD_Dataset", "images")
MAPS_ROOT  = os.path.join(PROJECT_ROOT, "BGD_Dataset", "maps")

MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "outputs", "models",
                               "braingaze_diffusion_model_v2.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS         = 50    # 10 warmup + 40 joint
WARMUP_EPOCHS  = 10
BATCH_SIZE     = 8

# ── Loss lambda weights ──────────────────────────────────────────────────────
LAMBDA_VICREG  = 3.0   # Variance regularisation   (was 2.0 in v1)
LAMBDA_DISC    = 2.0   # EEG discrimination loss   (new in v2)
LAMBDA_AUX_EEG = 0.5   # EEG-only auxiliary loss   (new in v2)


# ═══════════════════════════════════════════════════════════════════════════════
#  Loss Functions
# ═══════════════════════════════════════════════════════════════════════════════

def kld_loss(pred, target):
    """Kullback-Leibler divergence: KL(target ‖ softmax(pred))."""
    eps = 1e-7
    p = torch.softmax(pred.view(pred.size(0), -1), dim=1)
    t = target.view(target.size(0), -1)
    return torch.sum(t * (torch.log(t + eps) - torch.log(p + eps)), dim=1).mean()


def cc_loss(pred, target):
    """Negative Pearson Correlation Coefficient (minimise → maximise correlation)."""
    p = pred.view(pred.size(0), -1)
    t = target.view(target.size(0), -1)
    p = (p - p.mean(1, keepdim=True)) / (p.std(1, keepdim=True) + 1e-7)
    t = (t - t.mean(1, keepdim=True)) / (t.std(1, keepdim=True) + 1e-7)
    return -(p * t).mean(dim=1).mean()


def saliency_loss(pred, target):
    """Combined KLD + 0.5·CC  — standard saliency evaluation composite."""
    return kld_loss(pred, target) + 0.5 * cc_loss(pred, target)


def vicreg_variance_loss(eeg_feat):
    """
    VICReg variance term.
    Penalises the EEG encoder when all embeddings collapse to a similar vector.
    Forces the encoder to produce discriminative, per-trial representations.
    """
    if eeg_feat.size(0) <= 1:
        return torch.tensor(0.0, device=eeg_feat.device)
    std = torch.std(eeg_feat, dim=0, unbiased=False)   # (embed_dim,)
    return torch.mean(torch.relu(1.0 - std))


def eeg_discrimination_loss(pred_real, pred_shuffled):
    """
    EEG Discrimination Loss  (new in v2 — the primary bypass fix).

    For a given batch of images, two predictions are computed:
      • pred_real     — model(real_eeg,     image)
      • pred_shuffled — model(shuffled_eeg, image)   [same images, different brains]

    This loss penalises the model if the two saliency maps are too similar
    (measured by per-sample Pearson CC).  When CC > margin → positive loss
    → gradients push the FiLM parameters to make EEG content matter.

    The margin (0.85) is a soft target: we allow some correlation (different
    subjects still agree on salient regions) but penalise identical maps.
    """
    p = pred_real.view(pred_real.size(0), -1)
    q = pred_shuffled.view(pred_shuffled.size(0), -1)

    p_n = (p - p.mean(1, keepdim=True)) / (p.std(1, keepdim=True) + 1e-7)
    q_n = (q - q.mean(1, keepdim=True)) / (q.std(1, keepdim=True) + 1e-7)

    per_sample_cc = (p_n * q_n).mean(dim=1)   # (B,) ∈ [−1, 1]

    # Hinge: penalise if similarity exceeds the margin
    margin = 0.85
    return torch.relu(per_sample_cc - margin).mean()


def eeg_encoder_grad_norm(model):
    """Computes L2 gradient norm of just the EEG encoder — training signal monitor."""
    total_sq = 0.0
    for p in model.eeg_encoder.parameters():
        if p.grad is not None:
            total_sq += p.grad.data.norm(2).item() ** 2
    return total_sq ** 0.5


# ═══════════════════════════════════════════════════════════════════════════════
#  Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

def build_optimizer(model, eeg_lr, dec_lr):
    """Creates a fresh AdamW with separate LRs for EEG encoder vs. decoder/FiLM."""
    eeg_params = (list(model.eeg_encoder.parameters())
                  + list(model.subject_embed.parameters()))
    decoder_film_params = [
        p for name, p in model.named_parameters()
        if "eeg_encoder" not in name and "subject_embed" not in name
        and p.requires_grad
    ]
    return optim.AdamW(
        [{'params': eeg_params,          'lr': eeg_lr},
         {'params': decoder_film_params,  'lr': dec_lr}],
        weight_decay=1e-4
    )


def save_progression_image(model, track_eeg, track_image, track_subject,
                            track_sample, epoch, out_dir):
    """
    Saves a 3-panel PNG: Original | Joint prediction | EEG-only prediction.
    Shows how much the EEG branch contributes independently at each epoch.
    """
    model.eval()
    with torch.no_grad():
        pred_joint    = model(track_eeg, track_image, track_subject,
                              zero_image=False, apply_spatial_dropout=False)
        pred_eeg_only = model(track_eeg, track_image, track_subject,
                              zero_image=True,  apply_spatial_dropout=False)

    def to_map(t):
        m = t.squeeze().cpu().numpy()
        return (m - m.min()) / (m.max() - m.min() + 1e-8)

    orig = track_sample['image'].permute(1, 2, 0).numpy()
    orig = np.clip(orig * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax in axes:
        ax.axis('off')

    axes[0].imshow(orig)
    axes[0].set_title("Original Stimulus", fontsize=12)

    axes[1].imshow(orig)
    axes[1].imshow(to_map(pred_joint), cmap='jet', alpha=0.55)
    axes[1].set_title(f"Joint Prediction  (Epoch {epoch})", fontsize=12)

    axes[2].imshow(orig)
    axes[2].imshow(to_map(pred_eeg_only), cmap='jet', alpha=0.55)
    axes[2].set_title(f"EEG-Only Prediction  (Epoch {epoch})", fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"epoch_{epoch}.png"),
                bbox_inches='tight', dpi=130)
    plt.close()


def train():
    # ── Output directories ───────────────────────────────────────────────────
    CHECKPOINT_DIR  = os.path.join(PROJECT_ROOT, "outputs", "models",      "checkpoints_v2")
    PROGRESSION_DIR = os.path.join(PROJECT_ROOT, "outputs", "epoch_progression_v2")
    LOG_PATH        = os.path.join(PROJECT_ROOT, "outputs", "logs",        "train_log_v2.txt")
    for d in [CHECKPOINT_DIR, PROGRESSION_DIR,
              os.path.dirname(MODEL_SAVE_PATH), os.path.dirname(LOG_PATH)]:
        os.makedirs(d, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("Loading BrainGaze-Diffusion v2 matched dataset...")
    dataset    = EEGSaliencyDataset(EEG_DIR, STIM_ROOT, MAPS_ROOT, split="training")
    train_size = int(0.85 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

    # ── Model ────────────────────────────────────────────────────────────────
    model = BrainGazeDiffusionModel().to(DEVICE)

    # Fixed sample for visual tracking across epochs
    track_sample  = val_ds[0]
    track_eeg     = track_sample['eeg'].unsqueeze(0).to(DEVICE)
    track_image   = track_sample['image'].unsqueeze(0).to(DEVICE)
    track_subject = torch.tensor([track_sample['subject_id']]).to(DEVICE)

    # ── Phase 1 optimiser  (aggressive EEG warmup LR) ────────────────────────
    optimizer = build_optimizer(model, eeg_lr=2e-3, dec_lr=5e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float('inf')
    log_lines     = ["epoch,phase,train_loss,val_loss,disc_loss,aux_eeg_loss,eeg_grad_norm,lr"]

    print(f"\nStarting BrainGaze-Diffusion v2 on device: {DEVICE}")
    print(f"  Epochs       : {EPOCHS}  ({WARMUP_EPOCHS} warmup + {EPOCHS-WARMUP_EPOCHS} joint)")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  λ_VICReg     : {LAMBDA_VICREG}")
    print(f"  λ_Disc       : {LAMBDA_DISC}")
    print(f"  λ_AuxEEG     : {LAMBDA_AUX_EEG}")

    for epoch in range(EPOCHS):
        is_warmup = (epoch < WARMUP_EPOCHS)

        # ── Phase transition ─────────────────────────────────────────────────
        if epoch == WARMUP_EPOCHS:
            print("\n" + "═" * 70)
            print("  Phase transition → Joint Training.  Resetting optimiser.")
            print("═" * 70)
            # Clear stale warmup momentum; start fresh for joint phase
            optimizer = build_optimizer(model, eeg_lr=1e-3, dec_lr=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=EPOCHS - WARMUP_EPOCHS)

        phase_str = "Warmup" if is_warmup else "Joint"
        print(f"\n── Epoch {epoch+1:>2}/{EPOCHS}  [{phase_str}] " + "─" * 40)

        # ════════════════════════════════════════════════════════════════════
        #  TRAINING
        # ════════════════════════════════════════════════════════════════════
        model.train()
        run_sal  = 0.0
        run_disc = 0.0
        run_aux  = 0.0
        run_tot  = 0.0

        loop = tqdm(train_loader, desc="[Train]", leave=False)
        for batch in loop:
            eeg         = batch['eeg'].to(DEVICE)
            image       = batch['image'].to(DEVICE)
            target      = batch['saliency'].to(DEVICE)
            subject_ids = batch['subject_id'].to(DEVICE)

            optimizer.zero_grad()

            # ─────────────────────────────────────────────────────────────
            #  PHASE 1 — EEG-only warmup
            # ─────────────────────────────────────────────────────────────
            if is_warmup:
                out      = model(eeg, image, subject_ids,
                                 zero_image=True, apply_spatial_dropout=False)
                loss_sal = saliency_loss(out, target)

                eeg_feat  = model.get_eeg_features(eeg, subject_ids)
                loss_var  = vicreg_variance_loss(eeg_feat)

                loss = loss_sal + LAMBDA_VICREG * loss_var
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                run_sal += loss_sal.item()
                run_tot += loss.item()
                loop.set_postfix(sal=f"{loss_sal.item():.3f}",
                                 var=f"{loss_var.item():.3f}")

            # ─────────────────────────────────────────────────────────────
            #  PHASE 2 — Joint training with discrimination + auxiliary EEG
            # ─────────────────────────────────────────────────────────────
            else:
                # Pass A — main forward: real EEG + real image
                out_real = model(eeg, image, subject_ids,
                                 zero_image=False, apply_spatial_dropout=True)
                loss_sal = saliency_loss(out_real, target)

                # Pass B — discrimination: same images but shuffled EEG
                #   Shuffle the EEG/subject mapping within the batch so each
                #   sample sees a different subject's brain signal paired with
                #   its original image. The model must produce a DIFFERENT map.
                perm         = torch.randperm(eeg.size(0), device=DEVICE)
                shuffled_eeg = eeg[perm]
                shuffled_sub = subject_ids[perm]

                with torch.no_grad():
                    # No grad for shuffled branch — we only back-prop through
                    # pred_real so the gradient signal is clean and directed.
                    out_shuffled = model(shuffled_eeg, image, shuffled_sub,
                                        zero_image=False, apply_spatial_dropout=False)

                loss_disc = eeg_discrimination_loss(out_real, out_shuffled)

                # Pass C — EEG-only auxiliary: zero image, real EEG
                #   Forces the EEG encoder + FiLM path to independently
                #   approximate the target saliency without visual scaffolding.
                out_eeg_only = model(eeg, image, subject_ids,
                                     zero_image=True, apply_spatial_dropout=False)
                loss_aux = saliency_loss(out_eeg_only, target)

                # VICReg: keep EEG embeddings spread across the embedding space
                eeg_feat  = model.get_eeg_features(eeg, subject_ids)
                loss_var  = vicreg_variance_loss(eeg_feat)

                # Combined loss
                loss = (loss_sal
                        + LAMBDA_DISC    * loss_disc
                        + LAMBDA_AUX_EEG * loss_aux
                        + LAMBDA_VICREG  * loss_var)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                run_sal  += loss_sal.item()
                run_disc += loss_disc.item()
                run_aux  += loss_aux.item()
                run_tot  += loss.item()

                loop.set_postfix(
                    sal=f"{loss_sal.item():.3f}",
                    disc=f"{loss_disc.item():.3f}",
                    aux=f"{loss_aux.item():.3f}",
                    var=f"{loss_var.item():.3f}",
                )

        # EEG encoder gradient norm — measured right after last backward
        eeg_gnorm = eeg_encoder_grad_norm(model)
        scheduler.step()

        # ════════════════════════════════════════════════════════════════════
        #  VALIDATION
        # ════════════════════════════════════════════════════════════════════
        model.eval()
        val_sal = 0.0
        with torch.no_grad():
            for batch in val_loader:
                eeg         = batch['eeg'].to(DEVICE)
                image       = batch['image'].to(DEVICE)
                target      = batch['saliency'].to(DEVICE)
                subject_ids = batch['subject_id'].to(DEVICE)

                out     = model(eeg, image, subject_ids,
                                zero_image=is_warmup, apply_spatial_dropout=False)
                val_sal += saliency_loss(out, target).item()

        n_train = len(train_loader)
        n_val   = len(val_loader)
        avg_train = run_tot  / n_train
        avg_val   = val_sal  / n_val
        avg_disc  = run_disc / n_train
        avg_aux   = run_aux  / n_train
        current_lr = scheduler.get_last_lr()[0]

        line = (f"Epoch [{epoch+1:>2}/{EPOCHS}] | {phase_str:<7} | "
                f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
                f"Disc: {avg_disc:.4f} | AuxEEG: {avg_aux:.4f} | "
                f"EEG∇: {eeg_gnorm:.5f} | LR: {current_lr:.6f}")
        print(line)
        log_lines.append(
            f"{epoch+1},{phase_str},{avg_train:.4f},{avg_val:.4f},"
            f"{avg_disc:.4f},{avg_aux:.4f},{eeg_gnorm:.5f},{current_lr:.6f}"
        )

        # ── Per-epoch checkpoint ─────────────────────────────────────────────
        ckpt = os.path.join(CHECKPOINT_DIR, f"braingaze_v2_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt)

        # ── Progression image ────────────────────────────────────────────────
        save_progression_image(model, track_eeg, track_image, track_subject,
                               track_sample, epoch + 1, PROGRESSION_DIR)

        # ── Best model ───────────────────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ──▶  New best model saved  (val={avg_val:.4f})")

    # ── Save training log ─────────────────────────────────────────────────────
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print("\n" + "═" * 60)
    print("  BrainGaze v2 Training Complete!")
    print(f"  Best validation loss : {best_val_loss:.4f}")
    print(f"  Model saved to       : {MODEL_SAVE_PATH}")
    print("═" * 60)


if __name__ == "__main__":
    train()
