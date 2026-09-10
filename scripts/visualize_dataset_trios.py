import sys
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import shutil

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from src.eeg_saliency_pipeline import EEGSaliencyDataset

EEG_DIR = os.path.join(PROJECT_ROOT, "BGD_Dataset", "EEG")
STIM_ROOT = os.path.join(PROJECT_ROOT, "BGD_Dataset", "images")
MAPS_ROOT = os.path.join(PROJECT_ROOT, "BGD_Dataset", "maps")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "dataset_viz")
HTML_FILE = os.path.join(ARTIFACTS_DIR, "dataset_trios_verification.html")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def plot_eeg_channels(eeg_data, save_path):
    """
    Plots the preprocessed EEG signal waveforms for all 32 channels.
    """
    plt.figure(figsize=(8, 5))
    time = np.linspace(0, 500, eeg_data.shape[1]) # 500 ms window
    # Plot top 12 channels to keep it visually clean but representative
    channels_to_plot = min(12, eeg_data.shape[0])
    
    # Custom color palette for nice premium look
    colors = plt.cm.plasma(np.linspace(0, 0.8, channels_to_plot))
    
    for i in range(channels_to_plot):
        plt.plot(time, eeg_data[i] + i * 35, label=f"Ch {i+1}", color=colors[i], alpha=0.9, linewidth=1.2)
        
    plt.title("EEG Signal Amplitude (Sample Channels)", fontsize=10, color="#ffffff")
    plt.xlabel("Time (ms)", fontsize=8, color="#aaaaaa")
    plt.ylabel("µV (stacked offsets)", fontsize=8, color="#aaaaaa")
    plt.xticks(color="#888888")
    plt.yticks([]) # Hide Y tick labels for clean offsets
    plt.grid(True, linestyle="--", alpha=0.1)
    
    # Transparent dark background for premium visual alignment
    fig = plt.gcf()
    fig.patch.set_facecolor("#16161a")
    ax = plt.gca()
    ax.set_facecolor("#16161a")
    ax.spines['bottom'].set_color('#444444')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    
    plt.tight_layout()
    plt.savefig(save_path, facecolor="#16161a", edgecolor='none', dpi=100)
    plt.close()

def generate_trio_report():
    print("Generating Dataset Trio (EEG + Image + Gaze Map) Verification Report...")
    
    # Load training dataset split
    dataset = EEGSaliencyDataset(EEG_DIR, STIM_ROOT, MAPS_ROOT, split="training")
    
    html_content = """
    <html>
    <head>
        <title>Thesis Dataset Verification: Trios</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; background: #0c0c0e; color: #eee; padding: 40px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .card { background: #16161a; border-radius: 16px; padding: 24px; margin-bottom: 40px; border: 1px solid #2d2d35; }
            .grid { display: grid; grid-template-columns: 1fr 1fr 1.2fr; gap: 24px; align-items: center; }
            img { width: 100%; border-radius: 10px; border: 1px solid #333; }
            .label { font-size: 0.85em; color: #888; text-transform: uppercase; margin-bottom: 8px; font-weight: 600; letter-spacing: 0.5px; }
            h1 { font-weight: 300; font-size: 2.2em; text-align: center; margin-bottom: 5px; }
            p.subtitle { text-align: center; color: #666; margin-bottom: 40px; font-size: 1.0em; }
            .coco-badge { background: #7289da; color: white; padding: 4px 10px; border-radius: 8px; font-size: 0.75em; float: right; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>BrainGaze Dataset Aligned Trios</h1>
            <p class="subtitle">Verification of Stimulus Image, Human Gaze Ground Truth, and Preprocessed EEG Waves</p>
    """
    
    # Pick 10 spaced samples to show diversity
    indices = list(range(0, min(100, len(dataset)), 10)) if len(dataset) > 10 else list(range(len(dataset)))
    
    for i in indices:
        sample = dataset[i]
        coco_id = sample['coco_id']
        eeg_np = sample['eeg'].numpy()
        
        # File naming
        stim_fn = f"trio_stim_{coco_id}.jpg"
        gaze_fn = f"trio_gaze_{coco_id}.png"
        eeg_fn = f"trio_eeg_{coco_id}.png"
        
        # 1. Save EEG wave plot
        plot_eeg_channels(eeg_np, os.path.join(ARTIFACTS_DIR, eeg_fn))
        
        # 2. Copy/save stimulus image
        found_stim_path = None
        for split in ['train', 'val']:
            s_name = f"COCO_{split}2014_{int(coco_id):012d}.jpg"
            p = os.path.join(STIM_ROOT, s_name)
            if os.path.exists(p):
                found_stim_path = p
                break
                
        if found_stim_path:
            shutil.copy(found_stim_path, os.path.join(ARTIFACTS_DIR, stim_fn))
        else:
            # Reconstruct from dataset tensor if not found in SALICON split
            img_tensor = sample['image'].permute(1, 2, 0).numpy()
            img_tensor = img_tensor * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_tensor = np.clip(img_tensor * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(img_tensor).save(os.path.join(ARTIFACTS_DIR, stim_fn))
            
        # 3. Save gaze map heatmap
        gaze_map = sample['saliency'].squeeze().numpy()
        gaze_map = (gaze_map - gaze_map.min()) / (gaze_map.max() - gaze_map.min() + 1e-8)
        plt.imsave(os.path.join(ARTIFACTS_DIR, gaze_fn), gaze_map, cmap='jet')
        
        html_content += f"""
        <div class="card">
            <h2>
                Sample index #{i}
                <span class="coco-badge">COCO ID: {coco_id}</span>
            </h2>
            <div class="grid">
                <div>
                    <div class="label">1. Stimulus Image</div>
                    <img src="{stim_fn}">
                </div>
                <div>
                    <div class="label">2. Human Gaze Map</div>
                    <img src="{gaze_fn}">
                </div>
                <div>
                    <div class="label">3. EEG Waveform (12 channels)</div>
                    <img src="{eeg_fn}">
                </div>
            </div>
        </div>
        """
        
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(HTML_FILE, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Complete Trio verification report saved successfully at: {HTML_FILE}")

if __name__ == "__main__":
    generate_trio_report()
