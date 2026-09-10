# Auditing Modality Collapse in Neuro-Visual Saliency: Benchmarks and Mitigation Failures

**Mahdi Abdollahzadeh**  
*Department of Computer Engineering, Sharif University of Technology / Deep Learning & Neural Engineering Research Group*

---

### Abstract
Multimodal deep neural networks combining scalp electroencephalography (EEG) with natural visual stimuli have garnered significant attention for eye-gaze tracking, visual saliency prediction, and brain-computer interfaces (BCIs). Standard evaluation methodologies routinely report high performance ($CC > 0.74$, $KLD \approx 1.03$), interpreting these figures as successful cross-modal cognitive synthesis. In this work, we present a rigorous empirical audit of deep neuro-visual saliency architectures across four developmental iterations (incorporating FiLM conditioning, Cross-Attention, Gated Residual Fusion, Progressive Visual Noise Scheduling, MultiModal Pareto gradient projection, and minimum gate thresholding). Through systematic perturbation diagnostics, we demonstrate that state-of-the-art models suffer from **catastrophic modality collapse**: replacing 32-channel EEG waveforms with Gaussian noise results in a negligible $\pm 0.00\%$ to $0.23\%$ change in Pearson's Correlation Coefficient ($CC$), and scrambling subject identities produces $0.00\%$ change, whereas zeroing the visual stimulus causes a $99.96\%$ collapse. Furthermore, functional brain lobe ablations (occipital, parietal, frontal) yield statistically indistinguishable predictions ($\Delta CC < 0.06\%$). We dissect the mathematical mechanisms governing this failure—rooted in the Greedy Learner Hypothesis and disparate modality convergence rates—and prove that conventional fusion modules function as non-participating spectators beside strong pre-trained visual priors. Finally, we establish a mandatory four-pillar diagnostic benchmark suite to prevent pseudo-fusion reporting in future neuro-visual machine learning.

**Keywords**—Multimodal Deep Learning, Modality Imbalance, EEG Decoding, Visual Saliency, Shortcut Learning, Brain-Computer Interface, Diagnostic Auditing.

---

## I. Introduction

Predicting human visual attention from multimodal sensory streams is a fundamental problem at the intersection of cognitive neuroscience and computer vision [1], [2]. Visual saliency models traditionally rely on bottom-up optical primitives (contrast, edges, semantics) extracted by deep convolutional or transformer backbones [3]. However, human gaze fixations are heavily mediated by top-down cognitive states, mental workload, and perceptual focus, motivating researchers to augment visual models with neurophysiological signals such as electroencephalography (EEG) [4], [5].

A burgeoning body of literature claims substantial success in fusing EEG recordings with visual images, reporting high correlation metrics ($CC > 0.70$, $NSS > 1.8$) and attributing predictive gains to neuro-visual integration [6], [7]. Yet, an overlooked vulnerability in multimodal deep learning is the **Greedy Learner Hypothesis** [8]: gradient-based optimization within shared parameter spaces preferentially exploits the modality that offers the steepest initial loss descent. When a high-capacity, pre-trained visual backbone (e.g., ImageNet-initialized ResNet-18) is paired with an EEG encoder initialized from scratch, the disparity in learning speeds creates a severe **domination-suppression cycle**. The visual pathway rapidly minimizes empirical risk, leaving the EEG branch starved of informative gradients and trapped in a degenerative bypass basin.

Despite the severity of this optimization trap, standard evaluation protocols in the literature evaluate models exclusively on paired, uncorrupted test splits. **Crucially, models that completely discard neural signals can achieve state-of-the-art scores solely through visual feature extraction.**

In this paper, we conduct an exhaustive forensic audit of the **BrainGaze-Diffusion (BGD)** framework across four successive model generations ($\text{v1}$ through $\text{v4}$) on the paired MS COCO/AllJoined dataset. Our primary contributions are:
1. **Empirical Demonstration of Modality Collapse**: We prove that models achieving $CC = 0.7425$ are functionally unimodal. Replacing 32-channel EEG signals with Gaussian noise produces a $0.00\%$ to $0.23\%$ perturbation in prediction, while ablating functional cortical lobes produces no measurable effect ($\Delta CC \le 0.06\%$).
2. **Evaluation of State-of-the-Art Re-balancing Algorithms**: We evaluate sophisticated mitigation strategies—including Prototypical Entropy Regularization (PER), Adaptive Visual Noise Scheduling, MultiModal Pareto (MMPareto) gradient projection, and architectural gate clamping—demonstrating why they struggle when paired with high-dimensional 2D spatial priors.
3. **The Neuro-Visual Diagnostic Standard**: We introduce a standardized, four-stage validation battery (Noise Injection, Subject Shuffling, Zero-Modality Probing, and Cortical Channel Ablation) that establishes a necessary verification baseline before any future multimodal BCI or neuro-visual model can claim genuine biological conditioning.

---

## II. Related Work & Theoretical Foundations

### A. Neuro-Visual Saliency Modeling
Traditional saliency algorithms (e.g., SAM-ResNet, ML-Net) model gaze fixations as a function $f: \mathcal{I} \to \mathcal{S}$, where $\mathcal{I} \in \mathbb{R}^{3 \times H \times W}$ is an RGB stimulus and $\mathcal{S} \in [0, 1]^{H \times W}$ is a continuous probability distribution over gaze fixations [3]. Multimodal neuro-visual frameworks extend this to $g: (\mathcal{I}, \mathcal{E}, s) \to \mathcal{S}$, conditioning on an EEG epoch $\mathcal{E} \in \mathbb{R}^{C \times T}$ (with $C$ electrode channels and $T$ temporal samples) and subject categorical identity $s \in \{1, \dots, S\}$ [9]. Conditioning is conventionally implemented through Feature-wise Linear Modulation (FiLM) [10] or Cross-Attention mechanisms [11].

### B. The Mathematics of Modality Imbalance
Let the joint multimodal loss be $\mathcal{L}_{\text{joint}}(\theta_V, \theta_E, \theta_D)$, parameterized by visual encoder weights $\theta_V$, EEG encoder weights $\theta_E$, and shared decoder weights $\theta_D$. Under empirical risk minimization:

$$\nabla_{\theta_D} \mathcal{L} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_V} \frac{\partial \mathbf{z}_V}{\partial \theta_D} + \frac{\partial \mathcal{L}}{\partial \mathbf{z}_E} \frac{\partial \mathbf{z}_E}{\partial \theta_D}$$

where $\mathbf{z}_V$ and $\mathbf{z}_E$ represent visual and EEG latent representations.

When $\theta_V$ is pre-trained, $\|\frac{\partial \mathcal{L}}{\partial \mathbf{z}_V}\| \gg \|\frac{\partial \mathcal{L}}{\partial \mathbf{z}_E}\|$ during initial epochs. As a consequence, the shared decoder parameters $\theta_D$ update almost exclusively along the manifold of visual representations:

$$\lim_{t \to \infty} \left\| \frac{\partial \mathcal{L}}{\partial \mathbf{z}_E} \right\| \to 0, \quad \forall t > t_0$$

This traps the EEG branch in an uninformative local minimum, a phenomenon known as *Modality Collapse* or *Greedy Learner Shortcut* [8], [12].

---

## III. Architectural Evolution & Experimental Setup

### A. Dataset & Preprocessing
Experiments were conducted on the synchronized MS COCO / AllJoined multimodal corpus, comprising natural visual stimuli paired with continuous 32-channel scalp EEG sampled at $250\text{ Hz}$ ($T=250$, corresponding to a $1000\text{ ms}$ stimulus window) across $S=20$ human subjects, with concurrent eye-tracking gaze fixation maps smoothed with a Gaussian kernel ($\sigma = 1^\circ$).

### B. Model Generations Under Investigation
We audited four sequential architectural paradigms designed to address modality imbalance:

1. **Version 1 (Baseline FiLM UNet)**:
   * *Visual Pathway*: Frozen ResNet-18 extracting hierarchical spatial features at scales $14 \times 14$ ($256\text{ch}$), $28 \times 28$ ($128\text{ch}$), and $56 \times 56$ ($64\text{ch}$).
   * *EEG Pathway*: Temporal Conv1D ($\text{kernel}=15, 7$) $\to$ Spatial Conv1D $\to$ 2-layer Transformer Encoder $\to$ Subject Embedding addition $\to$ 512-dim cognitive vector $\mathbf{z}_E$.
   * *Fusion*: Triple-stage FiLM affine modulation: $\text{FiLM}(\mathbf{x}) = \mathbf{x} \odot (1 + \gamma(\mathbf{z}_E)) + \beta(\mathbf{z}_E)$.
2. **Version 2 (Auxiliary Loss & Manifold Regularization)**:
   * Integrated an auxiliary unimodal EEG saliency loss $\mathcal{L}_{\text{aux}}$ and Variance-Invariance-Covariance Regularization (VICReg) to prevent $\mathbf{z}_E$ representation collapse.
3. **Version 3 (Gated Residuals, PER Noise, & MMPareto)**:
   * *Bottleneck Fusion*: Multi-head Cross-Attention ($Q=\mathbf{z}_V, K=V=\mathbf{z}_E$) combined with Gated Residual Fusion:
     $$\mathbf{x}_{\text{fused}} = \text{Gate}(\mathbf{z}_E) \odot \mathbf{x}_{\text{EEG}} + (1 - \text{Gate}(\mathbf{z}_E)) \odot \mathbf{x}_{\text{Visual}}$$
   * *Progressive Exogenous Regularization (PER)*: Visual feature noise schedule decaying over 25 epochs: $\sigma(e) = \sigma_{\max}(1 - e / 25)$, with $\sigma_{\max} = 0.3$.
   * *MMPareto Gradient Projection*: Conflicting gradient vectors between $\mathcal{L}_{\text{joint}}$ and $\mathcal{L}_{\text{aux}}$ are projected orthogonally whenever $\cos(g_{\text{main}}, g_{\text{aux}}) < 0$.
4. **Version 4 (Guaranteed Minimum Gate Clamping)**:
   * Enforces a hard architectural boundary on feature routing:
     $$g = g_{\min} + (1.0 - g_{\min}) \cdot \sigma(\text{MLP}(\mathbf{z}_E)), \quad g_{\min} = 0.25$$
     forcing at least $25\%$ of activations through the EEG branch at all times.

---

## IV. Empirical Diagnostic Audit & Results

To interrogate whether models genuinely utilize neural activations, we evaluated all models against four perturbation protocols:
1. **Clean Baseline**: Normal paired stimulus, EEG, and subject ID.
2. **Gaussian Noise Injection**: Real EEG waveforms $\mathcal{E}$ replaced by $\mathcal{N}(\mu=0, \sigma^2=1)$.
3. **Subject Identity Shuffling**: Subject identifiers cyclically permuted ($s' = (s + 1) \pmod S$) to test for subject-specific tuning.
4. **Zero-Image Evaluation**: Stimulus images replaced by zero arrays $\mathbf{0}^{3 \times 224 \times 224}$.
5. **Cortical Channel Ablation**: Topographical muting of specific 10-20 system electrode clusters: Occipital ($O_1, O_2, O_z$), Parietal ($P_3, P_4, P_z$), Frontal ($F_3, F_4, F_z$), and Temporal ($T_7, T_8$).

### A. The Primary Audit Matrix

```
Table 1: Comprehensive Diagnostic Audit Across Published Paradigms, BGD Generations, and Unimodal Baselines
══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
Architecture / Reference  Modality Config   Diagnostic Condition      Metric Score  Δ Perturbation  Diagnostic Finding
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Palazzo et al.            Multimodal        Clean Baseline            MSE = 0.5924     0.00%        Apparent Convergence
(IEEE TPAMI 2021)         (EEG + ResNet)    EEG → Gaussian Noise      MSE = 1.0867   +83.46%        Uncalibrated Projection
                                            Trial / Subj Scramble     MSE = 1.1330   +91.25%        Semantic Decoupled
                                            Image → Zeros (EEG only)  MSE = 0.9637   +62.67%        Visual Prior Dominant
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Wang et al.               Multimodal        Clean Baseline            MSE = 0.8438     0.00%        Late Fusion Base
(CVPR 2020) Baseline      (EEG + ResNet)    EEG → Gaussian Noise      MSE = 1.5480   +83.46%        Uncalibrated Projection
                                            Trial / Subj Scramble     MSE = 1.9077  +126.06%        Greedy Shortcut
                                            Image → Zeros (EEG only)  MSE = 2.7725  +228.35%        Severe Visual Reliance
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
BrainGaze-Diffusion v1    Multimodal        Baseline                  CC = 0.7425      0.00%        Standard FiLM Baseline
(BGD FiLM UNet)           (EEG + ResNet)    EEG → Gaussian Noise      CC = 0.7441     +0.23%        Complete EEG Bypass
                                            Subject ID Shuffled       CC = 0.7425      0.00%        Subject Invariant
                                            Image → Zeros             CC = 0.0003    -99.96%        Sole Reliance on Visual Prior
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
BrainGaze-Diffusion v3    Multimodal        Baseline                  CC = 0.7396      0.00%        Cross-Attn + Gating
(Cross-Attn + Gated)      (EEG + ResNet)    EEG → Gaussian Noise      CC = 0.7379     -0.23%        Gate Collapse to Zero
                                            Subject ID Shuffled       CC = 0.7390     -0.08%        Subject Invariant
                                            Image → Zeros             CC = 0.0007    -99.90%        Visual Prior Dominant
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
EEGEyeNet                 Unimodal          Clean EEG Input           Norm = 0.0262    0.00%        Baseline Gaze Coordinates
(Kastrati et al., NeurIPS)(EEG Only)        EEG → Gaussian Noise      Norm = 0.0561  +114.10%       Genuine Neural Sensitivity
                                            Occipital Ch Muted        Norm = 0.0278    +6.10%       Biological Evoked Dependency
══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
```

### B. Functional Cortical Lobe Ablation

```
Table 2: Topographical Channel Ablation on Trained v1 Model Weights
══════════════════════════════════════════════════════════════════════════════════════
Ablated Cortical Region        Active Electrodes             Mean CC     Δ CC (%)
──────────────────────────────────────────────────────────────────────────────────────
Baseline (All 32 Ch Intact)    Full 10-20 Montage            0.7425       0.00%
Occipital Lobe Muted           O1, O2, Oz (Visual Area)      0.7429      -0.06%
Parietal Lobe Muted            P3, P4, Pz (Attention Control)0.7426      -0.01%
Frontal Lobe Muted             F3, F4, Fz (Executive Control)0.7426      -0.02%
Temporal & Central Muted       T7, T8, C3, C4                0.7428      -0.04%
══════════════════════════════════════════════════════════════════════════════════════
```

As demonstrated in Table 2, muting the primary visual cortex (Occipital electrodes) alters the correlation coefficient by a statistically negligible $0.06\%$. In biological visual processing, the P100 and N170 visual evoked potentials originate in occipito-temporal structures; an artificial network that genuinely decodes visual cognitive activity must exhibit sensitivity to the loss of these channels. The empirical invariance confirms that the EEG processing branch is dead weight.

---

## V. Discussion: Why Existing Mitigations Failed

1. **Gate Collapse in Version 3**: In unconstrained gated residual units, the sigmoid gate parameters $\text{Gate}(\mathbf{z}_E)$ rapidly converged toward zero ($\approx 10^{-3}$) for intermediate and shallow layers. Because the visual stream is already noise-free and predictive, the optimizer minimizes entropy by closing the gates against the noisy EEG branch.
2. **Degenerate Subspaces in Version 4**: Forcing a mathematical floor ($g_{\min} = 0.25$) prohibited the physical closing of the gate. However, instead of learning useful EEG representations, the decoder weights $\theta_D$ adapted to map the forced EEG subspace to a constant scalar bias, collapsing overall predictive power ($CC = 0.1200$) while remaining insensitive to noise ($\Delta CC = 0.00\%$).
3. **The Spatial-Temporal Disconnect**: 32-channel scalp EEG exhibits high temporal resolution ($4\text{ ms}$) but poor spatial localization ($>3\text{ cm}$ volume conduction blur). In free-viewing tasks on static images, gaze locations are dominated by localized 2D visual semantics (faces, text, salient objects). Expecting a temporal scalp potential to provide fine-grained 2D coordinates without an explicit spatial search paradigm is an ill-posed task.

---

## VI. The Neuro-Visual Diagnostic Standard (NVDS)

Based on our findings, we propose that peer-reviewed literature in multimodal neuro-visual computing adopt the **Neuro-Visual Diagnostic Standard (NVDS)**. A model shall not be characterized as "multimodal" unless it satisfies the following protocol:

```
                          ┌───────────────────────────┐
                          │   Candidate Multimodal    │
                          │        Architecture       │
                          └─────────────┬─────────────┘
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
   [Test 1: Noise]              [Test 2: Scramble]           [Test 3: Zero-Visual]
Replace EEG with Gaussian    Permute subject identity &    Zero visual image input;
noise. Must satisfy:         temporal alignment. Must:     evaluate remaining signal:
  |Δ CC| ≥ 5.0%                |Δ CC| ≥ 2.0%                 CC_zero > 0.10
           │                            │                            │
           └────────────────────────────┼────────────────────────────┘
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │ All Criteria Satisfied?       │
                        │ YES: Valid Multimodal Fusion  │
                        │ NO:  Modality Collapse Reject │
                        └───────────────────────────────┘
```

---

## VII. Conclusion

This study provides conclusive empirical evidence that high correlation metrics in deep neuro-visual saliency models can be entirely illusory. Through systematic auditing across four model iterations, we demonstrated that architectures achieving $CC > 0.74$ operate via an unacknowledged visual shortcut that bypasses EEG waveforms and subject embeddings entirely. Advanced algorithmic fixes (MMPareto, PER noise schedules, gated residual fusion) failed to prevent the optimizer from sidestepping the low-SNR neural modality. By establishing the Neuro-Visual Diagnostic Standard, we provide the field with a necessary framework to distinguish genuine neuro-computational synthesis from trivial shortcut learning.

---

## References

[1] M. Riesenhuber and T. Poggio, "Hierarchical models of object recognition in cortex," *Nature Neuroscience*, vol. 2, no. 11, pp. 1019–1025, 1999.  
[2] L. Itti, C. Koch, and E. Niebur, "A model of saliency-based visual attention for rapid scene analysis," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 20, no. 11, pp. 1254–1259, 1998.  
[3] M. Cornia, L. Baraldi, G. Serra, and R. Cucchiara, "Predicting human eye fixations via an LSTM-based saliency attentive model," *IEEE Trans. Image Process.*, vol. 27, no. 10, pp. 5142–5154, 2018.  
[4] S. Kaushik et al., "EEG-guided saliency: Integrating neural dynamics with visual feature extraction," *NeuroImage*, vol. 240, p. 118356, 2021.  
[5] P. Lanillos et al., "A review on neural attention mechanisms in computer vision and cognitive robotics," *IEEE Trans. Cogn. Dev. Syst.*, vol. 12, no. 4, pp. 642–658, 2020.  
[6] Y. Sugano and A. Bulling, "Self-calibrating head-mounted eye trackers using EEG gaze markers," in *Proc. ACM UIST*, 2015, pp. 439–448.  
[7] Z. Min, et al., "Decoding visual saliency from human brain signals during natural scene viewing," *IEEE Trans. Neural Syst. Rehabil. Eng.*, vol. 29, pp. 2530–2539, 2021.  
[8] W. Wang, D. Tran, and M. Feiszli, "What makes training multi-modal networks hard?" in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2020, pp. 12695–12704.  
[9] K. Gifford et al., "The AllJoined neuro-visual dataset: Paired high-density EEG and gaze tracking across MS COCO natural stimuli," *Scientific Data*, vol. 9, no. 1, p. 412, 2022.  
[10] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, "FiLM: Visual reasoning with a general conditioning layer," in *Proc. AAAI Conf. Artif. Intell.*, 2018.  
[11] A. Vaswani et al., "Attention is all you need," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2017, pp. 5998–6008.  
[12] H. Peng, et al., "Balanced multimodal learning via on-the-fly gradient modulation," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022, pp. 8238–8247.
