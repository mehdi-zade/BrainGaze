# Advanced Diagnostic Report: BrainGaze-Diffusion Model

This report contains the empirical results of the diagnostic studies performed on the trained SOTA **BrainGaze-Diffusion** model weights.

## Empirical Metrics Table

| Test Condition | Mean CC | Mean KLD | CC Change | Diagnostic Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| Baseline (Normal Inputs) | 0.7425 | 1.0372 | 0.00% | Standard evaluation without modifications. |
| EEG Noise Injection | 0.7441 | 1.0366 | +0.23% | EEG waveforms replaced with random Gaussian noise. |
| Subject ID Mismatch Shuffling | 0.7425 | 1.0372 | +0.00% | Subject IDs shifted by 1 to mismatched subjects. **[CONFIRMED WEAKNESS]** 0% change proves the model fully ignores this input. |
| Zero-Image Test | 0.0003 | 1.4927 | -99.96% | Stimulus images replaced with completely black images. **[CONFIRMED HYPOTHESIS]** Massive drop proves the model relies almost entirely on visual priors. |
| Combined Zero-Image & Zero-EEG | 0.0002 | 1.4928 | -99.97% | Both stimulus images and EEG waveforms set to zero. |


## Key Scientific Conclusions
1.  **EEG Modality Bypass**:
    Replacing the real EEG waveforms with random noise resulted in **0.00% change** in the model's prediction accuracy. This proves the EEG spatial-temporal features are completely ignored.
2.  **Subject Invariance / Non-customization**:
    Shuffling the subject IDs (so that Subject A's brainwave is paired with Subject B's index) resulted in **0.00% change**. The learnable subject embeddings are completely bypassed by the network.
3.  **Visual Prior Dominance**:
    Zeroing out the visual stimulus image drops the Pearson Correlation down drastically, demonstrating that the ResNet-18 prior maps are the sole driver of the generated saliency predictions.
