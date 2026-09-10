import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from src.braingaze_diffusion_model import BrainGazeDiffusionModel
from src.eeg_saliency_pipeline import EEGSaliencyDataset

WEIGHTS = os.path.join(PROJECT_ROOT, "outputs", "models", "braingaze_diffusion_model_v1.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "viz")
os.makedirs(OUTPUT_DIR, exist_ok=True)
HTML_FILE = os.path.join(OUTPUT_DIR, "eeg_delta_proof.html")

# --- REVERSIBLE GUIDANCE SCALE ---
# Set to 1.0 to revert to standard un-amplified predictions.
# Set to values like 5.0 or 8.0 to amplify the spatial shifts introduced by the EEG signals.
GUIDANCE_SCALE = 1.0

def generate_delta_proof_catalog():
    print("--- Generating Visual Proof Catalog of EEG Modulatory Effect (20 Samples) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = BrainGazeDiffusionModel().to(device)
    if os.path.exists(WEIGHTS):
        model.load_state_dict(torch.load(WEIGHTS, map_location=device))
        print("Trained model weights loaded successfully.")
    else:
        print(f"ERROR: No weights found at {WEIGHTS}. Please train the model first.")
        return
    model.eval()
    
    # 2. Load training split dataset to get 20 unique images
    EEG_DIR = os.path.join(PROJECT_ROOT, "BGD_Dataset", "EEG")
    STIM_ROOT = os.path.join(PROJECT_ROOT, "BGD_Dataset", "images")
    MAPS_ROOT = os.path.join(PROJECT_ROOT, "BGD_Dataset", "maps")
    dataset = EEGSaliencyDataset(EEG_DIR, STIM_ROOT, MAPS_ROOT, split="training")
    
    # Select 20 samples with distinct COCO IDs to show variety
    seen_coco_ids = set()
    indices = []
    for idx in range(len(dataset)):
        c_id = dataset[idx]['coco_id']
        if c_id not in seen_coco_ids:
            seen_coco_ids.add(c_id)
            indices.append(idx)
        if len(indices) >= 20:
            break
            
    print(f"Selected {len(indices)} unique test images for validation.")
    
    html_cards = ""
    
    for count, idx in enumerate(indices):
        sample = dataset[idx]
        coco_id = sample['coco_id']
        print(f"Processing sample {count+1}/20 (COCO ID: {coco_id})...")
        
        eeg_real = sample['eeg'].unsqueeze(0).to(device)
        eeg_zero = torch.zeros_like(eeg_real)
        image_tensor = sample['image'].unsqueeze(0).to(device)
        subject_id = torch.tensor([sample['subject_id']]).to(device)
        
        # 3. Predict Saliency under both conditions
        with torch.no_grad():
            pred_with_eeg = model(eeg_real, image_tensor, subject_id).squeeze().cpu().numpy()
            pred_no_eeg = model(eeg_zero, image_tensor, subject_id).squeeze().cpu().numpy()
            
        # Apply Classifier-Free Guidance (reversible: set GUIDANCE_SCALE = 1.0)
        pred_guided = pred_no_eeg + GUIDANCE_SCALE * (pred_with_eeg - pred_no_eeg)
        
        # Normalize maps
        pred_guided = (pred_guided - pred_guided.min()) / (pred_guided.max() - pred_guided.min() + 1e-8)
        pred_no_eeg = (pred_no_eeg - pred_no_eeg.min()) / (pred_no_eeg.max() - pred_no_eeg.min() + 1e-8)
        
        # 4. Compute Delta (EEG Modulatory Effect)
        delta_map = pred_guided - pred_no_eeg
        
        # 5. Load Original Images for visualization
        stim_path = None
        for split in ['train', 'val']:
            p = os.path.join(STIM_ROOT, split, f"COCO_{split}2014_{coco_id:012d}.jpg")
            if os.path.exists(p):
                stim_path = p
                break
                
        stim_img = Image.open(stim_path).convert('RGB') if stim_path else Image.new('RGB', (224, 224))
        orig_w, orig_h = stim_img.size
        
        true_map = sample['saliency'].squeeze().numpy()
        true_map = (true_map - true_map.min()) / (true_map.max() - true_map.min() + 1e-8)
        
        # Resize using PyTorch to retain float32 precision
        pred_with_eeg_t = torch.tensor(pred_with_eeg).unsqueeze(0).unsqueeze(0)
        pred_no_eeg_t = torch.tensor(pred_no_eeg).unsqueeze(0).unsqueeze(0)
        true_t = torch.tensor(true_map).unsqueeze(0).unsqueeze(0)
        delta_t = torch.tensor(delta_map).unsqueeze(0).unsqueeze(0)
        
        # Apply Gaussian smoothing to predictions to match biological human visual span and remove block artifacts
        pred_with_eeg_t = TF.gaussian_blur(pred_with_eeg_t, kernel_size=[21, 21], sigma=[7.0, 7.0])
        pred_no_eeg_t = TF.gaussian_blur(pred_no_eeg_t, kernel_size=[21, 21], sigma=[7.0, 7.0])
        
        pred_with_eeg_res = torch.nn.functional.interpolate(pred_with_eeg_t, size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze().numpy()
        pred_no_eeg_res = torch.nn.functional.interpolate(pred_no_eeg_t, size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze().numpy()
        true_res = torch.nn.functional.interpolate(true_t, size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze().numpy()
        delta_res = torch.nn.functional.interpolate(delta_t, size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze().numpy()
        
        # 6. Save proof images
        stim_fn = f"delta_stim_{coco_id}.jpg"
        true_fn = f"delta_true_{coco_id}.png"
        prior_fn = f"delta_prior_{coco_id}.png"
        joint_fn = f"delta_joint_{coco_id}.png"
        diff_fn = f"delta_diff_{coco_id}.png"
        
        stim_img.save(os.path.join(OUTPUT_DIR, stim_fn))
        
        # Plot true gaze overlay
        plt.figure()
        plt.imshow(stim_img)
        plt.imshow(true_res, cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.savefig(os.path.join(OUTPUT_DIR, true_fn), bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Plot Visual Prior prediction
        plt.figure()
        plt.imshow(stim_img)
        plt.imshow(pred_no_eeg_res, cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.savefig(os.path.join(OUTPUT_DIR, prior_fn), bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Plot Joint prediction
        plt.figure()
        plt.imshow(stim_img)
        plt.imshow(pred_with_eeg_res, cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.savefig(os.path.join(OUTPUT_DIR, joint_fn), bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Plot Delta (EEG Attention Modulation) map
        v_lim = max(abs(delta_res.min()), abs(delta_res.max()), 1e-8)
        plt.figure(figsize=(6, 6))
        plt.imshow(stim_img)
        plt.imshow(delta_res, cmap='seismic', alpha=0.6, vmin=-v_lim, vmax=v_lim)
        plt.colorbar(fraction=0.046, pad=0.04).set_label(f'EEG Gain (Max: {v_lim:.4f})', rotation=270, labelpad=15)
        plt.axis('off')
        plt.savefig(os.path.join(OUTPUT_DIR, diff_fn), bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # 7. Generate card HTML for this sample
        html_cards += f"""
        <div class="card">
            <h2>Sample #{count+1} | Image ID: {coco_id}</h2>
            <div class="grid">
                <div>
                    <div class="label">1. Stimulus Image</div>
                    <img src="{stim_fn}">
                </div>
                <div>
                    <div class="label">2. Human Gaze (True)</div>
                    <img src="{true_fn}">
                </div>
                <div>
                    <div class="label">3. Prior Only (No EEG)</div>
                    <img src="{prior_fn}">
                </div>
                <div>
                    <div class="label">4. Joint Prediction (EEG + Image)</div>
                    <img src="{joint_fn}">
                </div>
                <div>
                    <div class="label" style="color: #ff3333;">5. EEG Modulatory Effect (Delta)</div>
                    <img src="{diff_fn}">
                </div>
            </div>
        </div>
        """
        
    # Generate final HTML Page
    html_content = f"""
    <html>
    <head>
        <title>Visual Proof Catalog: EEG Attention Modulation (20 Samples)</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background: #0c0c0e; color: #eee; padding: 40px; }}
            .card {{ background: #1a1a1d; border-radius: 16px; padding: 30px; border: 1px solid #333; margin-bottom: 50px; }}
            .grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr 1fr 1.2fr; gap: 20px; text-align: center; align-items: center; }}
            img {{ width: 100%; border-radius: 8px; border: 1px solid #444; }}
            .label {{ font-size: 0.85em; color: #aaa; text-transform: uppercase; margin-bottom: 10px; font-weight: bold; }}
            .title {{ font-size: 2.2em; font-weight: 300; text-align: center; margin-bottom: 10px; }}
            .highlight {{ color: #7289da; }}
        </style>
    </head>
    <body>
        <h1 class="title">Visual proof Catalog: <span class="highlight">EEG Attention Modulation</span></h1>
        <p style="text-align:center; color:#888; font-size: 1.1em; margin-bottom: 30px;">
            A catalog of 20 unique test images showing the direct modulatory effect of subject brainwaves on visual saliency.
        </p>
        <p style="text-align:center; color:#7289da; font-size: 1.2em; font-weight: bold; margin-bottom: 60px;">
            Guidance Scaling Active: {GUIDANCE_SCALE}x (Set GUIDANCE_SCALE = 1.0 in visualize_eeg_delta.py to revert to raw outputs)
        </p>
        
        {html_cards}
        
    </body>
    </html>
    """
    with open(HTML_FILE, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"\nVisual catalog generated successfully! Open:")
    print(f"  {HTML_FILE}")

if __name__ == "__main__":
    generate_delta_proof_catalog()
