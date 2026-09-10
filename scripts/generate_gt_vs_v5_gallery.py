"""
generate_gt_vs_v5_gallery.py
=============================
Generates comprehensive, publication-grade Ground Truth vs. BrainGaze v5
Reconstruction comparisons across multiple representative test scenes.
"""

import os
import sys
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from src.braingaze_v5 import BrainGaze_v5_CVMR
from src.eeg_saliency_pipeline import EEGSaliencyDataset

ARTIFACTS_DIR = r"C:\Users\Mahdi Abdollahzadeh\.gemini\antigravity-ide\brain\ab740f41-977a-423e-8830-b9e02eb3f385"
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def calc_cc(p, g):
    p_norm = (p - np.mean(p)) / (np.std(p) + 1e-8)
    g_norm = (g - np.mean(g)) / (np.std(g) + 1e-8)
    return float(np.mean(p_norm * g_norm))

def calc_sim(p, g):
    p_sum = p / (np.sum(p) + 1e-8)
    g_sum = g / (np.sum(g) + 1e-8)
    return float(np.sum(np.minimum(p_sum, g_sum)))

def calc_kld(p, g):
    p_sum = p / (np.sum(p) + 1e-8)
    g_sum = g / (np.sum(g) + 1e-8)
    return float(np.sum(g_sum * np.log(1e-8 + g_sum / (p_sum + 1e-8))))

def denormalize_img(img_tensor):
    img_np = img_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    img_np = np.clip(img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
    return img_np

def overlay_heatmap(rgb_img, saliency_map, cmap='jet', alpha=0.55):
    norm_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
    cm = plt.get_cmap(cmap)
    colored_map = cm(norm_map)[:, :, :3] # RGB
    blended = (1 - alpha) * rgb_img + alpha * colored_map
    return np.clip(blended, 0, 1)

def main():
    print("Generating Ground Truth vs. BrainGaze v5 Reconstruction Gallery...")
    device = torch.device("cpu")

    # Load Model
    model = BrainGaze_v5_CVMR(num_basis=8).to(device)
    model_path = os.path.join(PROJECT_ROOT, "outputs", "models", "braingaze_v5_best.pth")
    checkpoint = torch.load(model_path, map_location=device)
    sd = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(sd)
    model.eval()

    # Load Test and Train Datasets
    eeg_dir = os.path.join(PROJECT_ROOT, "BGD_Dataset", "EEG")
    stim_root = os.path.join(PROJECT_ROOT, "BGD_Dataset", "images")
    maps_root = os.path.join(PROJECT_ROOT, "BGD_Dataset", "maps")
    ds_test = EEGSaliencyDataset(eeg_dir, stim_root, maps_root, split="test")
    ds_train = EEGSaliencyDataset(eeg_dir, stim_root, maps_root, split="training")

    # Select 4 representative scenes
    # 2 from test set (unseen holdout), 2 diverse training scenes
    candidates = [
        {"title": "Example 1: Sports & Action (Test Set, Subj 9)", "dataset": ds_test, "idx": 0, "tag": "test_sports_action"},
        {"title": "Example 2: People & Context (Test Set, Subj 10)", "dataset": ds_test, "idx": 200, "tag": "test_people_context"},
        {"title": "Example 3: Natural Scene & Animals (Validation Showcase)", "dataset": ds_train, "idx": 15, "tag": "train_natural_scene"},
        {"title": "Example 4: Indoor Object Arrangement (Validation Showcase)", "dataset": ds_train, "idx": 45, "tag": "train_indoor_objects"},
    ]

    saved_images = []

    # 1. Generate Individual Detailed Comparison Figures
    for item_idx, item in enumerate(candidates):
        sample = item["dataset"][item["idx"]]
        eeg = sample["eeg"].unsqueeze(0).to(device)
        img = sample["image"].unsqueeze(0).to(device)
        gt = sample["saliency"].squeeze().numpy()
        subj = torch.tensor([sample["subject_id"]], device=device)

        with torch.no_grad():
            pred, basis, weights, _, _ = model(eeg, img, subj)

        p = pred.squeeze().numpy()
        w = weights.squeeze().numpy()
        rgb = denormalize_img(img)

        # Normalize maps to [0, 1]
        gt_norm = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
        p_norm = (p - p.min()) / (p.max() - p.min() + 1e-8)
        diff_map = np.abs(gt_norm - p_norm)

        cc = calc_cc(p_norm, gt_norm)
        sim = calc_sim(p_norm, gt_norm)
        kld = calc_kld(p_norm, gt_norm)

        gt_overlay = overlay_heatmap(rgb, gt_norm, cmap='jet', alpha=0.52)
        v5_overlay = overlay_heatmap(rgb, p_norm, cmap='jet', alpha=0.52)

        fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))

        # Row 1: Visuals
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title(f"Input Stimulus (COCO #{sample['coco_id']})\nSubject #{sample['subject_id']}", fontsize=11, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(gt_overlay)
        axes[0, 1].set_title("Ground Truth Gaze Overlay\n(Human Eye-Tracker)", fontsize=11, fontweight='bold', color='navy')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(v5_overlay)
        axes[0, 2].set_title(f"BrainGaze v5 Reconstruction Overlay\n(Cognitive-Visual Routing)", fontsize=11, fontweight='bold', color='darkgreen')
        axes[0, 2].axis('off')

        # Row 2: Maps & Diagnostics
        axes[1, 0].imshow(gt_norm, cmap='jet')
        axes[1, 0].set_title("Ground Truth Saliency Density", fontsize=11, fontweight='bold')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(p_norm, cmap='jet')
        axes[1, 1].set_title(f"v5 Predicted Saliency Density\nCC: {cc:.4f} | SIM: {sim:.4f} | KLD: {kld:.4f}", fontsize=11, fontweight='bold', color='darkgreen')
        axes[1, 1].axis('off')

        # Absolute Error Map
        im_err = axes[1, 2].imshow(diff_map, cmap='inferno', vmin=0, vmax=1)
        axes[1, 2].set_title(f"Absolute Reconstruction Error\nMean Abs Diff: {diff_map.mean():.4f}", fontsize=11, fontweight='bold', color='maroon')
        axes[1, 2].axis('off')
        plt.colorbar(im_err, ax=axes[1, 2], fraction=0.046, pad=0.04)

        plt.suptitle(f"{item['title']}\nQuantitative Fidelity: CC = {cc:.4f}, SIM = {sim:.4f}, KLD = {kld:.4f}", fontsize=14, fontweight='bold', y=0.99)
        plt.tight_layout()

        single_out = os.path.join(FIGURES_DIR, f"comparison_{item['tag']}.png")
        plt.savefig(single_out, dpi=200, bbox_inches='tight')
        plt.close()

        # Copy to artifacts directory
        artifact_single = os.path.join(ARTIFACTS_DIR, f"comparison_{item['tag']}.png")
        shutil.copy(single_out, artifact_single)
        saved_images.append((single_out, artifact_single, item['title'], cc, sim, kld, sample['coco_id'], sample['subject_id']))
        print(f"Saved {single_out} (CC={cc:.4f})")

    # 2. Generate Master Side-by-Side 4-Row Panoramic Showcase Grid
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    col_headers = [
        "1. Input Stimulus (COCO)",
        "2. Human Ground Truth",
        "3. BrainGaze v5 Reconstruction",
        "4. Gaze Overlay on Stimulus",
        "5. Absolute Error Residual"
    ]

    for row_idx, item in enumerate(candidates):
        sample = item["dataset"][item["idx"]]
        eeg = sample["eeg"].unsqueeze(0).to(device)
        img = sample["image"].unsqueeze(0).to(device)
        gt = sample["saliency"].squeeze().numpy()
        subj = torch.tensor([sample["subject_id"]], device=device)

        with torch.no_grad():
            pred, basis, weights, _, _ = model(eeg, img, subj)

        p = pred.squeeze().numpy()
        rgb = denormalize_img(img)

        gt_norm = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
        p_norm = (p - p.min()) / (p.max() - p.min() + 1e-8)
        diff_map = np.abs(gt_norm - p_norm)
        v5_overlay = overlay_heatmap(rgb, p_norm, cmap='jet', alpha=0.52)

        cc = calc_cc(p_norm, gt_norm)
        sim = calc_sim(p_norm, gt_norm)

        row_visuals = [
            (rgb, None),
            (gt_norm, 'jet'),
            (p_norm, 'jet'),
            (v5_overlay, None),
            (diff_map, 'inferno')
        ]

        for col_idx in range(5):
            ax = axes[row_idx, col_idx]
            im_data, cmap = row_visuals[col_idx]
            if cmap is None:
                ax.imshow(im_data)
            else:
                ax.imshow(im_data, cmap=cmap, vmin=0 if cmap=='inferno' else None, vmax=1 if cmap=='inferno' else None)

            if row_idx == 0:
                ax.set_title(col_headers[col_idx], fontsize=12, fontweight='bold')

            ax.axis('off')

        # Add row title annotation on the left
        axes[row_idx, 0].text(-0.08, 0.5, f"Scene {row_idx+1}\nCOCO #{sample['coco_id']}\nSubj #{sample['subject_id']}\nCC={cc:.3f}",
                              transform=axes[row_idx, 0].transAxes, fontsize=10, fontweight='bold',
                              va='center', ha='right', bbox=dict(boxstyle='round,pad=0.4', facecolor='#e8f5e9', edgecolor='#2e7d32'))

    plt.suptitle("Comprehensive Benchmark: Human Eye-Tracking Ground Truth vs. BrainGaze v5 Reconstruction\nBilinear Modular Routing on Scalp EEG + Visual Foundations",
                 fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()

    master_out = os.path.join(FIGURES_DIR, "master_gt_vs_v5_gallery.png")
    plt.savefig(master_out, dpi=250, bbox_inches='tight')
    plt.close()

    artifact_master = os.path.join(ARTIFACTS_DIR, "master_gt_vs_v5_gallery.png")
    shutil.copy(master_out, artifact_master)
    print(f"Saved Master Showcase: {master_out}")

    # Copy existing key project figures to artifacts directory as well
    for key_fig in ["fig_2_v5_basis_decomposition_showcase.png", "fig_8_cross_generation_visual_grid.png"]:
        src_path = os.path.join(FIGURES_DIR, key_fig)
        if os.path.exists(src_path):
            dst_path = os.path.join(ARTIFACTS_DIR, key_fig)
            shutil.copy(src_path, dst_path)
            print(f"Copied {key_fig} to artifacts.")

    # Copy epoch 45 progression
    ep45_src = os.path.join(PROJECT_ROOT, "outputs", "epoch_progression_v5", "epoch_45.png")
    if os.path.exists(ep45_src):
        ep45_dst = os.path.join(ARTIFACTS_DIR, "epoch_45_progression.png")
        shutil.copy(ep45_src, ep45_dst)
        print(f"Copied epoch_45.png to artifacts.")

if __name__ == "__main__":
    main()
