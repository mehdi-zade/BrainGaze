# BGD Dataset (BrainGaze-Diffusion Dataset)

The **BGD Dataset** contains 32-channel scalp electroencephalography (EEG) recordings perfectly aligned with natural stimulus images from MS-COCO and ground-truth 2D visual saliency maps.

## Dataset Structure

```
BGD_Dataset/
├── EEG/             # Preprocessed 32-channel EEG signals (.npy array per trial, shape: 32 x 250)
├── images/          # Natural visual stimulus images (.jpg / .png, 224 x 224 x 3)
└── maps/            # Ground truth human fixation saliency density maps (.npy / .png, 224 x 224)
```

## Data Split & Alignment

* **Total Samples**: 3,234 trios aligned across 20 human subjects.
* **Channels**: 32 EEG channels (Fp1, Fp2, Fz, F3, F4, C3, C4, Cz, P3, P4, Pz, O1, O2, Oz, etc.).
* **Sampling Window**: 1,000 ms post-stimulus presentation ($250 \text{ Hz} \times 1 \text{ s} = 250 \text{ time points}$).
* **Normalization**: Signal amplitudes normalized per channel ($\mu = 0, \sigma = 1$).

## Dataset Access & Setup

The raw multimodal dataset (~100 GB) contains 32-channel continuous EEG tensors, stimulus imagery, and fixation maps. Due to GitHub's repository size limits, the raw binary arrays (`.npy`, `.jpg`, `.png`) are hosted externally.

To run the training or diagnostic pipelines locally:
1. Obtain the dataset directory archives (`EEG/`, `images/`, `maps/`).
2. Place or symlink the three directories directly inside this `BGD_Dataset/` folder:
   ```
   BGD_Dataset/
   ├── EEG/
   ├── images/
   ├── maps/
   └── README.md
   ```

