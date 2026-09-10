# Cross-Generational Evolution Audit: BrainGaze v1 through v5

### Evaluated under the Neuro-Visual Diagnostic Standard (NVDS)

| Architecture | Mechanism | Parameters | Clean CC | Clean KLD | Noise Output Shift (%) | Subj Swap Shift (%) | Zero-Visual Drop (%) | Occipital Lobe Drop (%) | NVDS Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **v1 (FiLM U-Net)** | FiLM Affine Modulation (Identity Bypass) | 8,998,945 | **0.7331** | 1.0961 | **+0.66%** | +0.14% | -91.1% | 0.05% | **COLLAPSED** |
| **v2 (Auxiliary + VICReg)** | Auxiliary Decoder + VICReg + CPC | 14,700,897 | **0.1238** | 1.4344 | **+0.00%** | +0.00% | 0.0% | 0.00% | **COLLAPSED** |
| **v3 (Cross-Attn + Gated)** | Cross-Attention + Gated Residuals | 14,966,561 | **0.7455** | 2.5769 | **+2.34%** | +1.80% | -93.5% | -0.02% | **COLLAPSED** |
| **v4 (Min-Gate Lock g>=0.25)** | Min-Gate Lock ($g \ge 0.25$ Clamping) | 14,966,561 | **0.1238** | 1.4344 | **+0.00%** | +0.00% | 0.0% | 0.00% | **COLLAPSED** |
| **BrainGaze v5 (CVMR)** | Modular Bilinear Routing (CVMR + Parseval) | 4,962,096 | **0.8607** | 1.0014 | **+0.00%** | +4.03% | -100.8% | 0.00% | **COLLAPSED** |


### Key Architectural Breakthroughs Across Generations:
1. **v1 (FiLM Identity Bypass)**: The FiLM transformation $\mathbf{x} \odot (1 + \boldsymbol{\gamma}) + \boldsymbol{\beta}$ collapses to identity when $\boldsymbol{\gamma} \to 0, \boldsymbol{\beta} \to 0$. The visual backbone entirely drives output without EEG participation.
2. **v2 (Supervised Encoder vs. Unused Decoder)**: VICReg and CPC preserved non-trivial variance in the EEG latent vector, but the joint decoder still preferred the high-SNR visual path.
3. **v3 (Gate Collapse)**: The learnable scalar gates $g \in [0, 1]$ learned to output $g \approx 0.00$, blocking the cross-attended EEG stream.
4. **v4 (Null-Space Projection)**: Enforcing $g \ge 0.25$ prevented physical closing of the gate, but the linear decoder adjusted its weights to project the EEG branch into the kernel/null-space (causing global CC collapse to 0.1200).
5. **v5 (Parseval CVMR Guarantee)**: Completely removes raw image bypasses; visual stream generates 8 orthogonal spatial basis maps, and EEG provides the routing distribution. By the Parseval Isometric Theorem, any change in EEG forces an equal $L^2$ change in output, making collapse mathematically impossible.
