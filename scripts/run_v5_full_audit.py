"""
run_v5_full_audit.py
====================
Exhaustive Forensic Audit & 7-Pillar Stress-Testing Suite for BrainGaze v5 (Epoch 46).
Evaluates under the Neuro-Visual Diagnostic Standard (NVDS) using official test splits.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from src.braingaze_v5 import BrainGaze_v5_CVMR
from src.eeg_saliency_pipeline import EEGSaliencyDataset

# -----------------------------------------------------------------------------
# Metric Definitions
# -----------------------------------------------------------------------------

def compute_cc(pred, target, eps=1e-7):
    p = pred.view(pred.size(0), -1)
    t = target.view(target.size(0), -1)
    p_norm = (p - p.mean(1, keepdim=True)) / (p.std(1, keepdim=True) + eps)
    t_norm = (t - t.mean(1, keepdim=True)) / (t.std(1, keepdim=True) + eps)
    return (p_norm * t_norm).mean(dim=1).mean().item()

def compute_kld(pred, target, eps=1e-7):
    p = pred.view(pred.size(0), -1)
    p = p / (p.sum(1, keepdim=True) + eps)
    t = target.view(target.size(0), -1)
    t = t / (t.sum(1, keepdim=True) + eps)
    return torch.sum(t * (torch.log(t + eps) - torch.log(p + eps)), dim=1).mean().item()

def compute_sim(pred, target, eps=1e-7):
    p = pred.view(pred.size(0), -1)
    p = p / (p.sum(1, keepdim=True) + eps)
    t = target.view(target.size(0), -1)
    t = t / (t.sum(1, keepdim=True) + eps)
    return torch.sum(torch.min(p, t), dim=1).mean().item()

def compute_mse(pred, target):
    return torch.mean((pred - target) ** 2).item()

def compute_nss(pred, target, eps=1e-7):
    p = pred.view(pred.size(0), -1)
    p_norm = (p - p.mean(1, keepdim=True)) / (p.std(1, keepdim=True) + eps)
    t = target.view(target.size(0), -1)
    # Binary fixations approximated by top 5% saliency pixels
    threshold = torch.quantile(t, 0.95, dim=1, keepdim=True)
    fix_mask = (t >= threshold).float()
    nss = (p_norm * fix_mask).sum(dim=1) / (fix_mask.sum(dim=1) + eps)
    return nss.mean().item()

def compute_auc_judd(pred, target, num_thresholds=100):
    p = pred.view(pred.size(0), -1).detach().cpu().numpy()
    t = target.view(target.size(0), -1).detach().cpu().numpy()
    aucs = []
    for i in range(p.shape[0]):
        sal = p[i]
        gt = t[i]
        thresh = np.percentile(gt, 95)
        fix = (gt >= thresh).astype(bool)
        if fix.sum() == 0 or (~fix).sum() == 0:
            continue
        sal_fix = sal[fix]
        sal_nonfix = sal[~fix]
        # Fast ROC AUC via Mann-Whitney U rank comparison
        n1 = len(sal_fix)
        n2 = len(sal_nonfix)
        all_scores = np.concatenate([sal_fix, sal_nonfix])
        ranks = np.argsort(np.argsort(all_scores)) + 1
        r1 = np.sum(ranks[:n1])
        u1 = r1 - (n1 * (n1 + 1)) / 2.0
        auc = u1 / (n1 * n2)
        aucs.append(auc)
    return float(np.mean(aucs)) if aucs else 0.5


# -----------------------------------------------------------------------------
# Main Audit Execution
# -----------------------------------------------------------------------------

def audit_v5(model_path, model_name="BrainGaze v5", n_samples=320):
    print("=" * 80)
    print(f"AUDITING {model_name.upper()} UNDER THE NEURO-VISUAL DIAGNOSTIC STANDARD")
    print(f"Checkpoint Path: {model_path}")
    print("=" * 80)

    device = torch.device("cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Instantiate & load model
    model = BrainGaze_v5_CVMR(num_basis=8).to(device)
    raw = torch.load(model_path, map_location=device)
    if "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
        epoch_info = raw.get("epoch", "N/A")
        val_cc_info = raw.get("val_cc", "N/A")
    else:
        state_dict = raw
        epoch_info = "46"
        val_cc_info = "N/A"

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"Model successfully loaded. Checkpoint Epoch: {epoch_info} | Recorded Val CC: {val_cc_info}")

    # 2. Load dataset
    eeg_dir = os.path.join(PROJECT_ROOT, "BGD_Dataset", "EEG")
    stim_root = os.path.join(PROJECT_ROOT, "BGD_Dataset", "images")
    maps_root = os.path.join(PROJECT_ROOT, "BGD_Dataset", "maps")

    ds_test = EEGSaliencyDataset(eeg_dir, stim_root, maps_root, split="test")
    indices = list(range(min(n_samples, len(ds_test))))
    sub_ds = Subset(ds_test, indices)
    loader = DataLoader(sub_ds, batch_size=16, shuffle=False)

    print(f"Evaluating across {len(sub_ds)} held-out test samples ({len(loader)} batches of 16)...")

    # Arrays for metrics
    clean_ccs, clean_klds, clean_sims, clean_nsss, clean_mses, clean_aucs = [], [], [], [], [], []
    noise_ccs, noise_klds, noise_shifts = [], [], []
    swap_ccs, swap_shifts = [], []
    zero_img_ccs, zero_img_drops = [], []
    muted_occ_ccs, muted_par_ccs, muted_fro_ccs = [], [], []
    
    all_routing_weights = []
    all_basis_gram_matrices = []

    noise_sigmas = [0.1, 0.5, 1.0, 2.0, 5.0]
    sigma_shifts = {s: [] for s in noise_sigmas}
    sigma_ccs = {s: [] for s in noise_sigmas}

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            eeg = batch['eeg'].to(device)
            img = batch['image'].to(device)
            gt  = batch['saliency'].to(device)
            sub = batch['subject_id'].to(device)

            # --- Clean Condition ---
            pred_clean, basis_maps, routing_w, _, _ = model(eeg, img, sub)
            clean_ccs.append(compute_cc(pred_clean, gt))
            clean_klds.append(compute_kld(pred_clean, gt))
            clean_sims.append(compute_sim(pred_clean, gt))
            clean_nsss.append(compute_nss(pred_clean, gt))
            clean_mses.append(compute_mse(pred_clean, gt))
            clean_aucs.append(compute_auc_judd(pred_clean, gt))

            all_routing_weights.append(routing_w.cpu().numpy())

            # Basis Orthogonality: Pairwise cosine similarity / Gram matrix
            flat = basis_maps.view(basis_maps.size(0), 8, -1)
            flat_n = flat / (flat.norm(dim=-1, keepdim=True) + 1e-7)
            gram = torch.bmm(flat_n, flat_n.transpose(1, 2))
            all_basis_gram_matrices.append(gram.cpu().numpy())

            # --- Pillar 1 & NVDS Condition B: Standard Gaussian Noise (sigma=1.0) ---
            eeg_noise = torch.randn_like(eeg)
            pred_noise, _, _, _, _ = model(eeg_noise, img, sub)
            noise_ccs.append(compute_cc(pred_noise, gt))
            shift_noise = ((pred_clean - pred_noise).abs().mean() / (pred_clean.abs().mean() + 1e-7)).item() * 100.0
            noise_shifts.append(shift_noise)

            # --- Pillar 4: Adversarial Cross-Trial EEG Swap ---
            perm_eeg = eeg[torch.randperm(eeg.size(0))]
            pred_swap, _, _, _, _ = model(perm_eeg, img, sub)
            swap_ccs.append(compute_cc(pred_swap, gt))
            shift_swap = ((pred_clean - pred_swap).abs().mean() / (pred_clean.abs().mean() + 1e-7)).item() * 100.0
            swap_shifts.append(shift_swap)

            # --- Pillar 5: Gaussian Noise Sweep ---
            eeg_std = eeg.std()
            for s in noise_sigmas:
                s_eeg = eeg + s * eeg_std * torch.randn_like(eeg)
                pred_s, _, _, _, _ = model(s_eeg, img, sub)
                s_shift = ((pred_clean - pred_s).abs().mean() / (pred_clean.abs().mean() + 1e-7)).item() * 100.0
                sigma_shifts[s].append(s_shift)
                sigma_ccs[s].append(compute_cc(pred_s, gt))

            # --- Pillar 6: Cortical Lobe Ablations ---
            # Occipital (Ch 28, 29, 30 ~ Oz, O1, O2)
            muted_occ = eeg.clone()
            muted_occ[:, 28:31, :] = 0.0
            pred_occ, _, _, _, _ = model(muted_occ, img, sub)
            muted_occ_ccs.append(compute_cc(pred_occ, gt))

            # Parietal (Ch 17, 18, 19 ~ P3, P4, Pz)
            muted_par = eeg.clone()
            muted_par[:, 17:20, :] = 0.0
            pred_par, _, _, _, _ = model(muted_par, img, sub)
            muted_par_ccs.append(compute_cc(pred_par, gt))

            # Frontal (Ch 0, 1, 2 ~ Fp1, Fp2, Fz)
            muted_fro = eeg.clone()
            muted_fro[:, :3, :] = 0.0
            pred_fro, _, _, _, _ = model(muted_fro, img, sub)
            muted_fro_ccs.append(compute_cc(pred_fro, gt))

            # --- Pillar 7: Zero-Visual Image Blind Probing ---
            zero_img = torch.zeros_like(img)
            pred_zero, _, _, _, _ = model(eeg, zero_img, sub)
            z_cc = compute_cc(pred_zero, gt)
            zero_img_ccs.append(z_cc)

    # -------------------------------------------------------------------------
    # Aggregate Metrics
    # -------------------------------------------------------------------------
    mean_clean_cc = float(np.mean(clean_ccs))
    mean_clean_kld = float(np.mean(clean_klds))
    mean_clean_sim = float(np.mean(clean_sims))
    mean_clean_nss = float(np.mean(clean_nsss))
    mean_clean_mse = float(np.mean(clean_mses))
    mean_clean_auc = float(np.mean(clean_aucs))

    mean_noise_cc = float(np.mean(noise_ccs))
    delta_noise_cc = float(((mean_noise_cc - mean_clean_cc) / mean_clean_cc) * 100.0)
    mean_noise_shift = float(np.mean(noise_shifts))

    mean_swap_cc = float(np.mean(swap_ccs))
    mean_swap_shift = float(np.mean(swap_shifts))
    delta_swap_cc = float(((mean_swap_cc - mean_clean_cc) / mean_clean_cc) * 100.0)

    mean_zero_cc = float(np.mean(zero_img_ccs))
    zero_drop_pct = float(((mean_zero_cc - mean_clean_cc) / mean_clean_cc) * 100.0)

    # Gram Matrix Statistics
    stacked_gram = np.concatenate(all_basis_gram_matrices, axis=0) # (N, 8, 8)
    mean_gram = np.mean(stacked_gram, axis=0)
    mask = ~np.eye(8, dtype=bool)
    off_diag_sims = mean_gram[mask]
    mean_off_diag_sim = float(np.mean(off_diag_sims))
    max_off_diag_sim = float(np.max(off_diag_sims))

    # Routing Distribution Statistics
    stacked_weights = np.concatenate(all_routing_weights, axis=0) # (N, 8)
    routing_var_per_basis = np.var(stacked_weights, axis=0).tolist()
    mean_routing_var = float(np.mean(routing_var_per_basis))
    eps = 1e-8
    sample_entropies = -np.sum(stacked_weights * np.log(stacked_weights + eps), axis=1)
    mean_entropy = float(np.mean(sample_entropies))
    max_entropy = float(np.log(8.0)) # ~2.079 nats
    channel_weight_means = np.mean(stacked_weights, axis=0).tolist()

    # Cortical Knockout Deltas
    occ_delta_pct = float(((np.mean(muted_occ_ccs) - mean_clean_cc) / mean_clean_cc) * 100.0)
    par_delta_pct = float(((np.mean(muted_par_ccs) - mean_clean_cc) / mean_clean_cc) * 100.0)
    fro_delta_pct = float(((np.mean(muted_fro_ccs) - mean_clean_cc) / mean_clean_cc) * 100.0)

    sweep_results = {}
    for s in noise_sigmas:
        sweep_results[str(s)] = {
            "output_shift_pct": float(np.mean(sigma_shifts[s])),
            "cc": float(np.mean(sigma_ccs[s]))
        }

    # NVDS Verdict
    c1 = (mean_noise_shift >= 5.0)
    c2 = (mean_off_diag_sim < 0.45)
    c3 = (mean_routing_var > 0.0005 and mean_entropy > 1.2)
    c4 = (mean_clean_cc >= 0.85)
    c5 = (mean_zero_cc < 0.05)
    nvds_passed = (c1 and c2 and c3 and c4 and c5)

    results = {
        "model_name": model_name,
        "model_path": model_path,
        "epoch": epoch_info,
        "n_samples": len(sub_ds),
        "clean_metrics": {
            "CC": round(mean_clean_cc, 4),
            "KLD": round(mean_clean_kld, 4),
            "SIM": round(mean_clean_sim, 4),
            "NSS": round(mean_clean_nss, 4),
            "MSE": round(mean_clean_mse, 4),
            "AUC_Judd": round(mean_clean_auc, 4)
        },
        "pillar_1_noise_sensitivity": {
            "clean_cc": round(mean_clean_cc, 4),
            "noise_cc": round(mean_noise_cc, 4),
            "delta_cc_pct": round(delta_noise_cc, 2),
            "output_perturbation_shift_pct": round(mean_noise_shift, 2),
            "threshold_required": ">= 5.0%",
            "passed": c1
        },
        "pillar_2_basis_orthogonality": {
            "mean_off_diagonal_cosine_similarity": round(mean_off_diag_sim, 4),
            "max_off_diagonal_cosine_similarity": round(max_off_diag_sim, 4),
            "gram_matrix": np.round(mean_gram, 4).tolist(),
            "passed": c2
        },
        "pillar_3_router_entropy": {
            "mean_entropy_nats": round(mean_entropy, 3),
            "max_capacity_nats": round(max_entropy, 3),
            "entropy_utilization_pct": round((mean_entropy / max_entropy) * 100.0, 1),
            "mean_routing_variance": round(mean_routing_var, 6),
            "channel_weight_means": [round(w, 3) for w in channel_weight_means],
            "passed": c3
        },
        "pillar_4_cross_trial_swap": {
            "swap_cc": round(mean_swap_cc, 4),
            "delta_cc_pct": round(delta_swap_cc, 2),
            "output_saliency_shift_pct": round(mean_swap_shift, 2)
        },
        "pillar_5_noise_intensity_sweep": sweep_results,
        "pillar_6_cortical_knockout": {
            "occipital_delta_cc_pct": round(occ_delta_pct, 2),
            "parietal_delta_cc_pct": round(par_delta_pct, 2),
            "frontal_delta_cc_pct": round(fro_delta_pct, 2)
        },
        "pillar_7_zero_visual_probe": {
            "zero_image_cc": round(mean_zero_cc, 4),
            "zero_image_drop_pct": round(zero_drop_pct, 2),
            "passed": c5
        },
        "overall_nvds_passed": nvds_passed
    }

    # Print Formatted Report Table
    print("\n" + "-" * 80)
    print(f">> SUMMARY OF FORENSIC RESULTS: {model_name.upper()}")
    print("-" * 80)
    print(f"  Sample Size (N)                     : {len(sub_ds)} Held-Out Samples")
    print(f"  Clean Multimodal Accuracy (CC)      : {mean_clean_cc:.4f}")
    print(f"  Clean Divergence (KLD)              : {mean_clean_kld:.4f}")
    print(f"  Normalized Scanpath Saliency (NSS)  : {mean_clean_nss:.4f}")
    print(f"  Similarity Metric (SIM)             : {mean_clean_sim:.4f}")
    print(f"  AUC-Judd                            : {mean_clean_auc:.4f}")
    print("-" * 80)
    print(f"  [Pillar 1] EEG Noise Output Shift   : +{mean_noise_shift:.2f}% (NVDS Target: >= 5.0%) -> {'PASSED [OK]' if c1 else 'FAILED'}")
    print(f"  [Pillar 1] EEG Noise Delta CC       : {delta_noise_cc:+.2f}%")
    print(f"  [Pillar 2] Basis Gram Off-Diag Sim  : {mean_off_diag_sim:.4f} (Target: < 0.45)    -> {'PASSED [OK]' if c2 else 'FAILED'}")
    print(f"  [Pillar 3] Routing Entropy H(alpha) : {mean_entropy:.3f} / {max_entropy:.3f} nats -> {'PASSED [OK]' if c3 else 'FAILED'}")
    print(f"  [Pillar 4] Cross-Trial Swap Shift   : +{mean_swap_shift:.2f}% (Inter-Trial Personalization)")
    print(f"  [Pillar 6] Occipital Lobe Knockout  : {occ_delta_pct:+.2f}%")
    print(f"  [Pillar 6] Parietal Lobe Knockout   : {par_delta_pct:+.2f}%")
    print(f"  [Pillar 6] Frontal Lobe Knockout    : {fro_delta_pct:+.2f}%")
    print(f"  [Pillar 7] Zero-Image Drop          : {zero_drop_pct:.2f}% (CC={mean_zero_cc:.4f})  -> {'PASSED [OK]' if c5 else 'FAILED'}")
    print("-" * 80)
    print(f"  OVERALL NVDS CERTIFICATION          : {'PASSED [GENUINE MULTIMODAL SYNTHESIS]' if nvds_passed else 'FAILED'}")
    print("=" * 80 + "\n")

    return results


def main():
    best_weights = os.path.join(PROJECT_ROOT, "outputs", "models", "braingaze_v5_best.pth")
    final_weights = os.path.join(PROJECT_ROOT, "outputs", "models", "braingaze_v5.pth")

    output_json = os.path.join(PROJECT_ROOT, "outputs", "v5_exhaustive_audit_report.json")

    all_results = {}
    if os.path.exists(best_weights):
        all_results["v5_best"] = audit_v5(best_weights, "BrainGaze v5 Best Checkpoint (Epoch 45)", n_samples=320)
    
    if os.path.exists(final_weights):
        all_results["v5_final_epoch46"] = audit_v5(final_weights, "BrainGaze v5 Interrupted Model (Epoch 46)", n_samples=320)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"Complete audit results exported to: {output_json}")


if __name__ == "__main__":
    main()
