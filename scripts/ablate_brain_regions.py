import sys
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from src.braingaze_diffusion_model import BrainGazeDiffusionModel
from src.eeg_saliency_pipeline import EEGSaliencyDataset

EEG_DIR = os.path.join(PROJECT_ROOT, "BGD_Dataset", "EEG")
STIM_ROOT = os.path.join(PROJECT_ROOT, "BGD_Dataset", "images")
MAPS_ROOT = os.path.join(PROJECT_ROOT, "BGD_Dataset", "maps")
WEIGHTS = os.path.join(PROJECT_ROOT, "outputs", "models", "braingaze_diffusion_model_v1.pth")
OUTPUT_MD = os.path.join(PROJECT_ROOT, "outputs", "brain_ablation_results.md")

# Standard 32-channel mapping grouped by functional brain lobes:
# Occipital (Visual Processing), Parietal (Spatial Attention), Frontal (Executive Control), Central/Temporal (Motor/Memory)
LOBE_CHANNELS = {
    "Occipital Lobe (Visual Area)": [24, 25, 26, 27, 28, 29, 30, 31],
    "Parietal Lobe (Attention Control)": [16, 17, 18, 19, 20, 21, 22, 23],
    "Frontal Lobe (Executive Planning)": [0, 1, 2, 3, 4, 5, 6, 7],
    "Central & Temporal Lobes (Auditory/Sensory)": [8, 9, 10, 11, 12, 13, 14, 15]
}

def evaluate_ablation(model, loader, device, mute_indices=None):
    ccs = []
    klds = []
    p_eps = 1e-7
    
    for batch in loader:
        eeg = batch['eeg'].clone()
        img = batch['image'].to(device)
        target = batch['saliency'].to(device)
        sub = batch['subject_id'].to(device)
        
        # Ablation step: Zero out the specified channel indices
        if mute_indices is not None:
            eeg[:, mute_indices, :] = 0.0
            
        eeg = eeg.to(device)
        
        with torch.no_grad():
            out = model(eeg, img, sub)
            
        for o, t in zip(out, target):
            p = torch.softmax(o.view(-1), dim=0)
            tg = t.view(-1)
            
            # Compute CC
            p_mu, p_std = p.mean(), p.std()
            t_mu, t_std = tg.mean(), tg.std()
            p_norm = (p - p_mu) / (p_std + 1e-7)
            t_norm = (tg - t_mu) / (t_std + 1e-7)
            cc = (p_norm * t_norm).mean().item()
            ccs.append(cc)
            
            # Compute KLD
            kld = torch.sum(tg * (torch.log(tg + p_eps) - torch.log(p + p_eps))).item()
            klds.append(kld)
            
    return np.mean(ccs), np.mean(klds)

def run_ablation_study():
    print("--- Starting Neuro-Ablation Explainability Study ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = BrainGazeDiffusionModel().to(device)
    if os.path.exists(WEIGHTS):
        model.load_state_dict(torch.load(WEIGHTS, map_location=device))
        print("Loaded trained weights successfully.")
    else:
        print(f"ERROR: No trained weights found at {WEIGHTS}. Please train the model first.")
        return
        
    model.eval()
    
    # 2. Load test split dataset
    dataset = EEGSaliencyDataset(EEG_DIR, STIM_ROOT, MAPS_ROOT, split="test")
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    results = []
    
    # Baseline run (All channels intact)
    print("\nRunning baseline evaluation (All channels active)...")
    base_cc, base_kld = evaluate_ablation(model, loader, device, mute_indices=None)
    print(f"  Baseline CC: {base_cc:.4f} | KLD: {base_kld:.4f}")
    results.append({
        "Condition": "Baseline (All Channels Intact)",
        "Mean CC": f"{base_cc:.4f}",
        "Mean KLD": f"{base_kld:.4f}",
        "CC Drop": "0.00%",
        "Impact Level": "None"
    })
    
    # Ablate each lobe individually
    for lobe_name, indices in LOBE_CHANNELS.items():
        print(f"\nAblating {lobe_name} (Zeroing out channels {indices})...")
        cc, kld = evaluate_ablation(model, loader, device, mute_indices=indices)
        
        cc_drop_pct = ((base_cc - cc) / base_cc) * 100
        
        # Categorize impact
        if cc_drop_pct > 15:
            impact = "CRITICAL (Primary Driver)"
        elif cc_drop_pct > 5:
            impact = "MODERATE (Supporting Role)"
        else:
            impact = "NEGLIGIBLE (Low relevance)"
            
        print(f"  CC: {cc:.4f} | KLD: {kld:.4f} | Drop: {cc_drop_pct:.2f}%")
        results.append({
            "Condition": f"Muted {lobe_name}",
            "Mean CC": f"{cc:.4f}",
            "Mean KLD": f"{kld:.4f}",
            "CC Drop": f"{cc_drop_pct:.2f}%",
            "Impact Level": impact
        })
        
    # Generate Thesis-ready Markdown Report
    markdown_table = "| Condition | Mean CC | Mean KLD | CC Drop | Impact Level |\n"
    markdown_table += "| :--- | :--- | :--- | :--- | :--- |\n"
    for r in results:
        markdown_table += f"| {r['Condition']} | {r['Mean CC']} | {r['Mean KLD']} | {r['CC Drop']} | {r['Impact Level']} |\n"
    
    report_content = f"""# Explainable AI (XAI) Report: Brain Lobe Ablation Study

This study ablated different functional lobes of the brain during the test phase of the **BrainGaze-Diffusion** model to identify the physiological mechanism of visual attention.

## Channel Ablation Metrics Table

{markdown_table}

## Neuroscientific Interpretation of Results
1. **Critical Region Identification**: The region showing the largest **CC Drop %** represents the primary source of decodable visual attention signals. 
2. **Visual Processing vs. Spatial Attention**: 
   * A high drop in the **Occipital Lobe** suggests the model is relying heavily on raw sensory visual features encoded in the early visual cortex.
   * A high drop in the **Parietal Lobe** indicates the model is successfully decoding top-down attentional shifts mediated by the dorsal attention stream.
"""
    
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"\nAblation study complete! Thesis report saved to: {OUTPUT_MD}")

if __name__ == "__main__":
    run_ablation_study()
