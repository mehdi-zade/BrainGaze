"""
train_braingaze_fast_v4.py
===========================
10-epoch fast training script for BrainGaze v4.
Guarantees min_gate=0.25 to prevent EEG bypass.
"""

import sys, os, torch, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from src.braingaze_diffusion_model_v4 import BrainGazeDiffusionModel
from src.eeg_saliency_pipeline        import EEGSaliencyDataset

# ── Config ────────────────────────────────────────────────────────────────────
EEG_DIR   = os.path.join(PROJECT_ROOT, "BGD_Dataset", "EEG")
STIM_ROOT = os.path.join(PROJECT_ROOT, "BGD_Dataset", "images")
MAPS_ROOT = os.path.join(PROJECT_ROOT, "BGD_Dataset", "maps")
SAVE_PATH = os.path.join(PROJECT_ROOT, "outputs", "models",
                         "braingaze_diffusion_model_v4.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS        = 10
WARMUP_EPOCHS = 3
BATCH_SIZE    = 8

L_VICREG = 3.0
L_DISC   = 2.0
L_AUX    = 0.5


# ── Losses ────────────────────────────────────────────────────────────────────

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

def sal_loss(pred, target):
    return kld_loss(pred, target) + 0.5 * cc_loss(pred, target)

def vicreg_var(feat):
    if feat.size(0) <= 1:
        return torch.tensor(0.0, device=feat.device)
    return torch.mean(torch.relu(1.0 - torch.std(feat, dim=0, unbiased=False)))

def disc_loss(pred_real, pred_shuf):
    p = pred_real.view(pred_real.size(0), -1)
    q = pred_shuf.view(pred_shuf.size(0), -1)
    p = (p - p.mean(1, keepdim=True)) / (p.std(1, keepdim=True) + 1e-7)
    q = (q - q.mean(1, keepdim=True)) / (q.std(1, keepdim=True) + 1e-7)
    cc = (p * q).mean(dim=1)
    return torch.relu(cc - 0.85).mean()


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    CKPT_DIR = os.path.join(PROJECT_ROOT, "outputs", "models", "checkpoints_v4")
    PROG_DIR = os.path.join(PROJECT_ROOT, "outputs", "epoch_progression_v4")
    for d in [CKPT_DIR, PROG_DIR, os.path.dirname(SAVE_PATH)]:
        os.makedirs(d, exist_ok=True)

    print("Loading dataset...")
    ds = EEGSaliencyDataset(EEG_DIR, STIM_ROOT, MAPS_ROOT, split="training")
    tr_n = int(0.85 * len(ds))
    tr_ds, va_ds = random_split(ds, [tr_n, len(ds) - tr_n],
                                generator=torch.Generator().manual_seed(42))
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train: {len(tr_ds)} | Val: {len(va_ds)}")

    # Model v4
    model = BrainGazeDiffusionModel(
        noise_sigma=0.5, noise_decay_epochs=5, min_gate=0.25
    ).to(DEVICE)

    # Track sample
    ts = va_ds[0]
    t_eeg  = ts['eeg'].unsqueeze(0).to(DEVICE)
    t_img  = ts['image'].unsqueeze(0).to(DEVICE)
    t_sub  = torch.tensor([ts['subject_id']]).to(DEVICE)

    # Optimizer parameters
    eeg_p = list(model.eeg_encoder.parameters()) + list(model.subject_embed.parameters())
    dec_p = [p for n, p in model.named_parameters()
             if 'eeg_encoder' not in n and 'subject_embed' not in n and p.requires_grad]

    best_val = float('inf')

    print(f"\n{'='*60}")
    print(f"  Fast Training v4: {EPOCHS} epochs ({WARMUP_EPOCHS}W + {EPOCHS-WARMUP_EPOCHS}J)")
    print(f"  Device: {DEVICE}  |  min_gate: 0.25")
    print(f"{'='*60}\n")

    for epoch in range(EPOCHS):
        is_warmup = epoch < WARMUP_EPOCHS

        if epoch == 0:
            opt = optim.AdamW([
                {'params': eeg_p, 'lr': 3e-3},
                {'params': dec_p, 'lr': 1e-4},
            ], weight_decay=1e-4)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=WARMUP_EPOCHS)

        if epoch == WARMUP_EPOCHS:
            print(f"\n{'='*60}\n  → Phase 2: Joint Training\n{'='*60}")
            opt = optim.AdamW([
                {'params': eeg_p, 'lr': 2e-3},
                {'params': dec_p, 'lr': 2e-4},
            ], weight_decay=1e-4)
            sch = optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=EPOCHS - WARMUP_EPOCHS)

        if not is_warmup:
            model.visual_noise.set_epoch(epoch - WARMUP_EPOCHS)

        phase = "Warmup" if is_warmup else "Joint"
        sigma = model.visual_noise.current_sigma.item()
        print(f"\n── Epoch {epoch+1}/{EPOCHS} [{phase}] σ={sigma:.3f} ──")

        model.train()
        tot_loss = 0.0

        for batch in tqdm(tr_dl, desc="[Train]", leave=False):
            eeg   = batch['eeg'].to(DEVICE)
            img   = batch['image'].to(DEVICE)
            tgt   = batch['saliency'].to(DEVICE)
            subs  = batch['subject_id'].to(DEVICE)
            opt.zero_grad()

            if is_warmup:
                out = model(eeg, img, subs, zero_image=True, apply_visual_noise=False)
                loss = sal_loss(out, tgt)
                ef = model.get_eeg_features(eeg, subs)
                loss = loss + L_VICREG * vicreg_var(ef)
            else:
                out_r = model(eeg, img, subs,
                              zero_image=False, apply_visual_noise=True)
                l_sal = sal_loss(out_r, tgt)

                perm = torch.randperm(eeg.size(0), device=DEVICE)
                with torch.no_grad():
                    out_s = model(eeg[perm], img, subs[perm],
                                  zero_image=False, apply_visual_noise=False)
                l_disc = disc_loss(out_r, out_s)

                out_e = model(eeg, img, subs,
                              zero_image=True, apply_visual_noise=False)
                l_aux = sal_loss(out_e, tgt)

                ef = model.get_eeg_features(eeg, subs)
                l_var = vicreg_var(ef)

                loss = l_sal + L_DISC * l_disc + L_AUX * l_aux + L_VICREG * l_var

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += loss.item()

        sch.step()

        # Validate
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in va_dl:
                eeg  = batch['eeg'].to(DEVICE)
                img  = batch['image'].to(DEVICE)
                tgt  = batch['saliency'].to(DEVICE)
                subs = batch['subject_id'].to(DEVICE)
                out  = model(eeg, img, subs,
                             zero_image=is_warmup, apply_visual_noise=False)
                v_loss += sal_loss(out, tgt).item()

        avg_t = tot_loss / len(tr_dl)
        avg_v = v_loss / len(va_dl)
        print(f"  Train: {avg_t:.4f} | Val: {avg_v:.4f}")

        # Checkpoint
        torch.save(model.state_dict(),
                   os.path.join(CKPT_DIR, f"braingaze_v4_epoch_{epoch+1}.pth"))

        # Progression image
        with torch.no_grad():
            pj = model(t_eeg, t_img, t_sub, zero_image=False, apply_visual_noise=False)
            pe = model(t_eeg, t_img, t_sub, zero_image=True,  apply_visual_noise=False)

        def m2np(t):
            a = t.squeeze().cpu().numpy()
            return (a - a.min()) / (a.max() - a.min() + 1e-8)

        orig = ts['image'].permute(1, 2, 0).numpy()
        orig = np.clip(orig * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)

        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        for a in ax: a.axis('off')
        ax[0].imshow(orig); ax[0].set_title("Original")
        ax[1].imshow(orig); ax[1].imshow(m2np(pj), cmap='jet', alpha=0.55)
        ax[1].set_title(f"Joint (Ep {epoch+1})")
        ax[2].imshow(orig); ax[2].imshow(m2np(pe), cmap='jet', alpha=0.55)
        ax[2].set_title(f"EEG-Only (Ep {epoch+1})")
        plt.tight_layout()
        plt.savefig(os.path.join(PROG_DIR, f"epoch_{epoch+1}.png"),
                    bbox_inches='tight', dpi=100)
        plt.close()

        # Best model
        if avg_v < best_val:
            best_val = avg_v
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ──▶ Best model saved (val={avg_v:.4f})")

    print(f"\n{'='*60}")
    print(f"  Done! Best val: {best_val:.4f}")
    print(f"  Model: {SAVE_PATH}")
    print(f"  Now run:  python scripts/run_diagnostics_v4.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
