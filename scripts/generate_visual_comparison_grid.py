"""
generate_visual_comparison_grid.py
==================================
Generates side-by-side visual saliency prediction maps across all 5 generations:
[Image] [Ground Truth] [v1] [v2] [v3] [v4] [v5 BrainGaze]
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

import src.braingaze_diffusion_model as v1_mod
import src.braingaze_diffusion_model_v2 as v2_mod
import src.braingaze_diffusion_model_v3 as v3_mod
import src.braingaze_diffusion_model_v4 as v4_mod
from src.braingaze_v5 import BrainGaze_v5_CVMR
from src.eeg_saliency_pipeline import EEGSaliencyDataset

def main():
    print("Generating 5-Generation Qualitative Saliency Map Comparison Grid...")
    device = torch.device("cpu")

    models_dir = os.path.join(PROJECT_ROOT, "outputs", "models")
    m1 = v1_mod.BrainGazeDiffusionModel().to(device)
    m1.load_state_dict(torch.load(os.path.join(models_dir, "braingaze_diffusion_model_v1.pth"), map_location=device))
    m1.eval()

    m2 = v2_mod.BrainGazeDiffusionModel().to(device)
    m2.load_state_dict(torch.load(os.path.join(models_dir, "braingaze_diffusion_model_v2.pth"), map_location=device))
    m2.eval()

    m3 = v3_mod.BrainGazeDiffusionModel().to(device)
    m3.load_state_dict(torch.load(os.path.join(models_dir, "braingaze_diffusion_model_v3.pth"), map_location=device))
    m3.eval()

    m4 = v4_mod.BrainGazeDiffusionModel().to(device)
    m4.load_state_dict(torch.load(os.path.join(models_dir, "braingaze_diffusion_model_v4.pth"), map_location=device))
    m4.eval()

    m5 = BrainGaze_v5_CVMR(num_basis=8).to(device)
    raw_v5 = torch.load(os.path.join(models_dir, "braingaze_v5_best.pth"), map_location=device)
    sd_v5 = raw_v5["model_state_dict"] if "model_state_dict" in raw_v5 else raw_v5
    m5.load_state_dict(sd_v5)
    m5.eval()

    # Load 2 test samples: one with person/faces, one with natural objects
    eeg_dir = os.path.join(PROJECT_ROOT, "BGD_Dataset", "EEG")
    stim_root = os.path.join(PROJECT_ROOT, "BGD_Dataset", "images")
    maps_root = os.path.join(PROJECT_ROOT, "BGD_Dataset", "maps")
    ds = EEGSaliencyDataset(eeg_dir, stim_root, maps_root, split="test")

    test_indices = [4, 12] # Two representative scenes
    
    fig, axes = plt.subplots(len(test_indices), 7, figsize=(22, 6.5))

    col_titles = [
        "Input Stimulus (COCO)",
        "Human Gaze (Ground Truth)",
        "v1: FiLM U-Net (CC=0.733)",
        "v2: Auxiliary (CC=0.124)",
        "v3: Cross-Attn (CC=0.746)",
        "v4: Min-Gate Lock (CC=0.124)",
        "★ v5: BrainGaze CVMR (CC=0.861)"
    ]

    for row_idx, s_idx in enumerate(test_indices):
        sample = ds[s_idx]
        eeg = sample['eeg'].unsqueeze(0).to(device)
        img = sample['image'].unsqueeze(0).to(device)
        gt  = sample['saliency'].squeeze().numpy()
        sub = torch.tensor([sample['subject_id']], device=device)

        with torch.no_grad():
            p1 = m1(eeg, img, sub, zero_image=False).squeeze().cpu().numpy()
            p2 = m2(eeg, img, sub, zero_image=False, apply_spatial_dropout=False).squeeze().cpu().numpy()
            p3 = m3(eeg, img, sub, zero_image=False, apply_visual_noise=False).squeeze().cpu().numpy()
            p4 = m4(eeg, img, sub, zero_image=False, apply_visual_noise=False).squeeze().cpu().numpy()
            p5, _, _, _, _ = m5(eeg, img, sub)
            p5 = p5.squeeze().cpu().numpy()

        # Unnormalize RGB image
        img_np = img.squeeze().cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)

        row_images = [img_np, gt, p1, p2, p3, p4, p5]
        cmaps = [None, 'jet', 'jet', 'jet', 'jet', 'jet', 'jet']

        for col_idx in range(7):
            ax = axes[row_idx, col_idx]
            im_data = row_images[col_idx]
            if cmaps[col_idx] is None:
                ax.imshow(im_data)
            else:
                norm_im = (im_data - im_data.min()) / (im_data.max() - im_data.min() + 1e-8)
                ax.imshow(norm_im, cmap=cmaps[col_idx])
            
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontweight='bold', fontsize=11, color='darkgreen' if 'v5' in col_titles[col_idx] else 'black')
            ax.axis('off')

    plt.suptitle("Qualitative Visual Evolution: Cross-Generational Saliency Maps Across Developmental Stages", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    out_path = os.path.join(PROJECT_ROOT, "outputs", "figures", "fig_8_cross_generation_visual_grid.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated Figure 8: {out_path}")

if __name__ == "__main__":
    main()
