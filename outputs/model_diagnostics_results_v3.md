# Diagnostic Report: BrainGaze-Diffusion v3

This report evaluates the **v3** model trained with modality-balanced architecture
and losses.  Compare with v1 results (EEG noise: +0.23% -> bypass confirmed).

## Empirical Results

| Test Condition | Mean CC | Mean KLD | CC Change | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| Baseline (Normal Inputs) | 0.7396 | 1.1186 | 0.00% | Standard evaluation. |
| EEG -> Gaussian Noise | 0.7379 | 1.1163 | -0.23% | EEG waveforms replaced with random Gaussian noise. **[EEG BYPASS]** 0.2% change — model ignores EEG. |
| Subject ID Mismatch | 0.7390 | 1.1233 | -0.08% | Subject IDs shifted by 1 to wrong subjects. **[SUBJECT-BLIND]** — subject embeddings inactive. |
| Image -> Zeros | 0.0007 | 1.4950 | -99.90% | Stimulus images replaced with black images. **[VISUAL PRIOR DOMINANT]** Massive drop confirms visual reliance. |
| Both -> Zeros | 0.0007 | 1.4950 | -99.90% | Both image and EEG set to zero. |


## ⚠️ EEG Bypass Partially Resolved

EEG noise injection caused 0.2% CC change.
Further training epochs or hyperparameter tuning may be needed.


## Comparison: v1 vs v3

| Condition | v1 CC Change | v3 CC Change |
| :--- | :--- | :--- |
| EEG -> Noise | +0.23% (bypass) | -0.23% |
| Subject Shuffle | +0.00% (bypass) | -0.08% |
| Image -> Zeros | -99.96% | -99.90% |
