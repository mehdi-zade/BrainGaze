# Explainable AI (XAI) Report: Brain Lobe Ablation Study

This study ablated different functional lobes of the brain during the test phase of the **BrainGaze-Diffusion** model to identify the physiological mechanism of visual attention.

## Channel Ablation Metrics Table

| Condition | Mean CC | Mean KLD | CC Drop | Impact Level |
| :--- | :--- | :--- | :--- | :--- |
| Baseline (All Channels Intact) | 0.7425 | 1.0372 | 0.00% | None |
| Muted Occipital Lobe (Visual Area) | 0.7429 | 1.0370 | -0.06% | NEGLIGIBLE (Low relevance) |
| Muted Parietal Lobe (Attention Control) | 0.7426 | 1.0371 | -0.01% | NEGLIGIBLE (Low relevance) |
| Muted Frontal Lobe (Executive Planning) | 0.7426 | 1.0371 | -0.02% | NEGLIGIBLE (Low relevance) |
| Muted Central & Temporal Lobes (Auditory/Sensory) | 0.7428 | 1.0371 | -0.04% | NEGLIGIBLE (Low relevance) |


## Neuroscientific Interpretation of Results
1. **Critical Region Identification**: The region showing the largest **CC Drop %** represents the primary source of decodable visual attention signals. 
2. **Visual Processing vs. Spatial Attention**: 
   * A high drop in the **Occipital Lobe** suggests the model is relying heavily on raw sensory visual features encoded in the early visual cortex.
   * A high drop in the **Parietal Lobe** indicates the model is successfully decoding top-down attentional shifts mediated by the dorsal attention stream.
