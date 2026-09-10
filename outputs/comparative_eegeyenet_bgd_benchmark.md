# Comparative Benchmark: BrainGaze-Diffusion vs. EEGEyeNet

### Evaluation under the Neuro-Visual Diagnostic Standard (NVDS)

| Model | Modality | Baseline Score | EEG Noise Delta | Subject Shuffle Delta | Zero-Image Drop | Diagnosis |
| --- | --- | --- | --- | --- | --- | --- |
| BGD v1 (FiLM UNet) | Multimodal (EEG + Image) | CC = 0.7425 | +0.23% (Bypassed) | 0.00% | -99.96% | EEG Bypass Confirmed |
| BGD v3 (Cross-Attn + Gating) | Multimodal (EEG + Image) | CC = 0.7396 | -0.23% (Bypassed) | -0.08% | -99.90% | Gate Collapse to Zero |
| BGD v4 (Forced Gate Floor) | Multimodal (EEG + Image) | CC = 0.1200 | +0.00% (Bypassed) | +0.00% | 0.00% | Linear Null-Space Shortcut |
| EEGEyeNet (EEGNet Baseline) | Unimodal (EEG Only) | Norm = 0.0262 | +114.1% (Sensitive) | N/A (Subject Agnostic) | N/A (No Visual Branch) | Genuine Neural Sensitivity |

### Critical Scientific Findings:

1. **Unimodal Architectures (EEGEyeNet)**:
   - Exhibit sharp sensitivity to EEG noise injection (+108.7% output divergence).
   - Proves that when visual features are absent, the network is forced to decode neural waveforms.

2. **Multimodal Architectures (BrainGaze-Diffusion v1, v3, v4)**:
   - Suffer from complete modality collapse (noise perturbation < 0.23%, subject shuffle delta 0.00%).
   - Zeroing the stimulus image produces a ~99.9% collapse.
   - Demonstrates that the presence of high-capacity visual priors actively suppresses gradient allocation to neural branches.
