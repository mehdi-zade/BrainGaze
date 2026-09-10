import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append(os.getcwd())
from src.eeg_saliency_pipeline import EEGSaliencyDataset

# Paths
EEG_DIR = os.path.join(os.getcwd(), 'BGD_Dataset', 'EEG')
STIM_ROOT = os.path.join(os.getcwd(), 'BGD_Dataset', 'images')
MAPS_ROOT = os.path.join(os.getcwd(), 'BGD_Dataset', 'maps')

OUT_DIR_OUTPUTS = os.path.join(os.getcwd(), 'outputs', 'figures')
OUT_DIR_BRAIN = r'C:\Users\Mahdi Abdollahzadeh\.gemini\antigravity-ide\brain\ab740f41-977a-423e-8830-b9e02eb3f385'
os.makedirs(OUT_DIR_OUTPUTS, exist_ok=True)

# Load dataset
ds = EEGSaliencyDataset(EEG_DIR, STIM_ROOT, MAPS_ROOT, split='training')
sample = ds[0]

coco_id = sample['coco_id']
subject_id = sample['subject_id']
eeg_tensor = sample['eeg'] # [32, 250]
img_tensor = sample['image'] # [3, 224, 224]
sal_tensor = sample['saliency'] # [1, 224, 224]

# Un-normalize image tensor for plotting
mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
img_np = img_tensor.numpy() * std + mean
img_np = np.clip(img_np.transpose(1, 2, 0), 0, 1)

sal_np = sal_tensor.numpy().squeeze(0) # [224, 224]
eeg_np = eeg_tensor.numpy() # [32, 250]

# Standard 10-20 channel names for 32 channels
channel_names = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5",
    "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
    "TP9", "CP5", "CP1", "CP2", "CP6", "TP10", "P7", "P3",
    "Pz", "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10"
]

# Set up figure with clean white background for academic publication
fig = plt.figure(figsize=(17, 8.5), facecolor="#ffffff")
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.45], wspace=0.30)

# Academic Light Theme Colors
title_color = "#0f172a"
text_color = "#1e293b"
accent_cyan = "#0284c7"
accent_purple = "#7e22ce"
accent_amber = "#b45309"

# Title Banner
fig.suptitle(f"Multimodal BGD Dataset Trio Sample Showcase (COCO ID: #{coco_id} | Subject ID: #{subject_id:02d})", 
             fontsize=16, fontweight="bold", color=title_color, y=0.96)

# -------------------------------------------------------------
# 1. Visual Stimulus Image (MS COCO)
# -------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(img_np)
ax1.set_title("1. Visual Stimulus Image\n(MS COCO 2014 Benchmark)", fontsize=12, fontweight="bold", color=accent_cyan, pad=12)
ax1.axis("off")

ax1.text(0.5, -0.15, "Data Dimensions:\n3 × 224 × 224 pixels\n[RGB Channels × H × W]", 
         transform=ax1.transAxes, ha="center", va="top", fontsize=10.5, fontweight="bold", color=text_color,
         bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8fafc", edgecolor="#0284c7", linewidth=1.5, alpha=0.95))

# -------------------------------------------------------------
# 2. Human Attention Saliency Map (SALICON / Ground Truth)
# -------------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(img_np * 0.35 + 0.15) # Dim background
im_sal = ax2.imshow(sal_np, cmap="magma", alpha=0.85)
ax2.set_title("2. Human Attention Saliency Map\n(SALICON Eye-Tracking Ground Truth)", fontsize=12, fontweight="bold", color=accent_purple, pad=12)
ax2.axis("off")

# Colorbar placed neatly under the map
cbar_ax = fig.add_axes([0.375, 0.28, 0.22, 0.018])
cbar = plt.colorbar(im_sal, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Normalized Gaze Density", color="#334155", fontsize=8.5, labelpad=2, fontweight="bold")
cbar.ax.tick_params(labelsize=8, colors="#334155")

ax2.text(0.5, -0.15, "Data Dimensions:\n1 × 224 × 224 pixels\n[Continuous Saliency Map]", 
         transform=ax2.transAxes, ha="center", va="top", fontsize=10.5, fontweight="bold", color=text_color,
         bbox=dict(boxstyle="round,pad=0.6", facecolor="#faf5ff", edgecolor="#7e22ce", linewidth=1.5, alpha=0.95))

# -------------------------------------------------------------
# 3. Synchronized Brain Signals (Alljoined1 EEG Waveforms)
# -------------------------------------------------------------
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor("#f8fafc")

time_axis = np.linspace(0, 500, 250) # 500 ms window @ 500 Hz

# Select 12 representative channels
rep_indices = [0, 3, 4, 12, 13, 14, 23, 24, 25, 28, 29, 30]
colors_eeg = plt.cm.inferno(np.linspace(0.15, 0.85, len(rep_indices)))

# Standardize each channel waveform for clean stack display
offset_step = 6.0
y_ticks_pos = []
y_ticks_labels = []

for rank, idx in enumerate(rep_indices):
    raw_wave = eeg_np[idx]
    norm_wave = (raw_wave - np.mean(raw_wave)) / (np.std(raw_wave) + 1e-6)
    y_offset = (len(rep_indices) - 1 - rank) * offset_step
    
    ax3.plot(time_axis, norm_wave + y_offset, color=colors_eeg[rank], linewidth=1.4, alpha=0.95)
    y_ticks_pos.append(y_offset)
    y_ticks_labels.append(channel_names[idx])

ax3.set_xlim(0, 500)
ax3.set_ylim(-3.5, len(rep_indices) * offset_step + 3.5)

ax3.set_yticks(y_ticks_pos)
ax3.set_yticklabels(y_ticks_labels, color="#0f172a", fontsize=9, fontweight="bold")

ax3.set_xlabel("Time Window (ms) [0 ms to 500 ms @ 500 Hz Sampling]", fontsize=10, fontweight="bold", color=text_color, labelpad=8)
ax3.set_ylabel("32-Channel Scalp EEG Electrodes (10-20 System)", fontsize=10, fontweight="bold", color=text_color, labelpad=8)
ax3.set_title("3. Synchronized Brain Signals\n(Alljoined1 32-Channel Scalp EEG)", fontsize=12, fontweight="bold", color=accent_amber, pad=12)

ax3.tick_params(colors="#334155", labelsize=9)
ax3.grid(True, linestyle="--", alpha=0.4, color="#cbd5e1")

# ERP Window Shading
ax3.axvspan(90, 120, color="#38bdf8", alpha=0.25, label="P100 Visual ERP (90-120 ms)")
ax3.axvspan(280, 340, color="#c084fc", alpha=0.25, label="P300 Cognitive ERP (280-340 ms)")
ax3.legend(loc="upper right", facecolor="#ffffff", edgecolor="#cbd5e1", labelcolor="#0f172a", fontsize=8.5)

ax3.text(0.5, -0.15, "Data Dimensions:\n32 Channels × 250 Time Steps\n[32 Electrodes × 500 ms Window at 500 Hz]", 
         transform=ax3.transAxes, ha="center", va="top", fontsize=10.5, fontweight="bold", color=text_color,
         bbox=dict(boxstyle="round,pad=0.6", facecolor="#fffbeb", edgecolor="#b45309", linewidth=1.5, alpha=0.95))

plt.subplots_adjust(top=0.86, bottom=0.22, left=0.06, right=0.97)

# Save figure to both locations
fn = "trio_dataset_sample_showcase.png"
path1 = os.path.join(OUT_DIR_OUTPUTS, fn)
path2 = os.path.join(OUT_DIR_BRAIN, fn)

plt.savefig(path1, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
plt.savefig(path2, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()

print("Regenerated white-background 300 DPI showcase image.")
