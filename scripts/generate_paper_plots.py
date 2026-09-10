import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)

# Set IEEE-compatible typography and styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
    'figure.titlesize': 11,
    'text.usetex': False,
    'lines.linewidth': 1.5,
    'grid.alpha': 0.35,
    'grid.linestyle': '--'
})

def generate_comparative_audit_chart():
    """
    Figure: Sensitivity Divergence between Multimodal Bypass vs. Unimodal Sensitivity
    Compares % Metric Divergence under Gaussian Noise Substitution across:
    BGD v1, BGD v3, Palazzo et al., Wang et al., and EEGEyeNet
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={'width_ratios': [1.3, 1]})

    models = ['BGD v1\n(FiLM)', 'BGD v3\n(CrossAttn)', 'Palazzo et al.\n(TPAMI 2021)', 'Wang et al.\n(CVPR 2020)']
    x = np.arange(len(models))
    width = 0.35

    rects1 = ax1.bar(x - width/2, [0.23, 0.23, 1.2, 2.5], width, label='EEG Noise Shift (|Δ%|)', color='#2b5c8f', edgecolor='black', linewidth=0.8)
    rects2 = ax1.bar(x + width/2, [99.96, 99.90, 62.67, 228.35], width, label='Zero-Visual Drop (% Collapse)', color='#d9534f', edgecolor='black', linewidth=0.8)

    ax1.set_ylabel('Performance Shift / Drop (%)')
    ax1.set_title('(a) Multimodal Paradigms: Modality Imbalance', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, axis='y')
    ax1.set_ylim(0, 260)

    for rect in rects1[:2]:
        h = rect.get_height()
        ax1.annotate(f'{h:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#2b5c8f')

    paradigms = ['Multimodal\n(BGD / Palazzo)', 'Unimodal EEG\n(EEGEyeNet)']
    sensitivity = [0.23, 114.10]
    colors = ['#888888', '#2ca02c']

    bars2 = ax2.bar(paradigms, sensitivity, color=colors, edgecolor='black', width=0.45, linewidth=0.8)
    ax2.set_ylabel('Perturbation Response |Δ Metric| (%)')
    ax2.set_title('(b) Neural Pathway Sensitivity', fontweight='bold')
    ax2.grid(True, axis='y')
    ax2.set_ylim(0, 135)
    
    ax2.annotate('Bypassed\n(0.23%)', xy=(0, 0.23), xytext=(0, 20), textcoords="offset points",
                 ha='center', va='bottom', fontsize=8, fontweight='bold', color='#555555',
                 arrowprops=dict(arrowstyle='->', lw=0.8, color='#555555'))

    ax2.annotate('Genuine Sensitivity\n(+114.10%)', xy=(1, 114.10), xytext=(0, 5), textcoords="offset points",
                 ha='center', va='bottom', fontsize=8, fontweight='bold', color='#2ca02c')

    ax2.axhline(5.0, color='red', linestyle=':', linewidth=1.2, label='NVDS Sensitivity Threshold (5%)')
    ax2.legend(loc='upper left', framealpha=0.9, fontsize=7.5)

    plt.tight_layout()
    pdf_path = os.path.join("outputs", "figures", "fig_multimodal_collapse_audit.pdf")
    png_path = os.path.join("outputs", "figures", "fig_multimodal_collapse_audit.png")
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {pdf_path} and {png_path}")

def generate_spatial_temporal_disconnect():
    """
    Figure: The Spatial-Temporal Disconnect
    Visualizes why scalp EEG (high temporal, blurred spatial) gets bypassed
    by high-resolution convolutional feature maps.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.5))

    modalities = ['Scalp EEG\n(Electrode Dipoles)', 'Visual Prior\n(ResNet-18)']
    spatial_res = [35.0, 1.0]
    colors = ['#e67e22', '#3498db']

    ax1.barh(modalities, spatial_res, color=colors, edgecolor='black', height=0.45)
    ax1.set_xlabel('Spatial Blur Radius (mm) [Lower is sharper]')
    ax1.set_title('(a) Spatial Precision Disparity', fontweight='bold')
    ax1.grid(True, axis='x')
    for i, v in enumerate(spatial_res):
        ax1.text(v + 1, i, f'~{v:.0f} mm' if v > 5 else f'< 1 mm', va='center', fontweight='bold', fontsize=8)

    epochs = np.linspace(0, 20, 100)
    visual_loss = np.exp(-0.4 * epochs) * 0.9 + 0.1
    eeg_loss = np.exp(-0.03 * epochs) * 0.95 + 0.05

    ax2.plot(epochs, visual_loss, label=r'Visual Pathway $\nabla_{\theta_V} \mathcal{L}$', color='#2980b9', lw=2)
    ax2.plot(epochs, eeg_loss, label=r'Neural Pathway $\nabla_{\theta_E} \mathcal{L}$', color='#e67e22', lw=2, linestyle='--')
    ax2.set_xlabel('Optimization Steps (Epochs)')
    ax2.set_ylabel('Modality Empirical Loss')
    ax2.set_title('(b) Gradient Starvation Dynamics', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True)

    plt.tight_layout()
    pdf_path = os.path.join("outputs", "figures", "fig_spatial_disconnect.pdf")
    png_path = os.path.join("outputs", "figures", "fig_spatial_disconnect.png")
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {pdf_path} and {png_path}")

if __name__ == "__main__":
    generate_comparative_audit_chart()
    generate_spatial_temporal_disconnect()
