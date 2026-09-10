import torch
import torch.nn as nn
import torchvision.models as models


# =============================================================================
#  BrainGaze-Diffusion Model  —  Version 2
#
#  Changelog vs. v1  (braingaze_diffusion_model.py):
#
#  [FIX 1] Modality dropout REMOVED from model.forward().
#          In v1, a 50% random coin flip inside the model zeroed visual features
#          during joint training. This was non-deterministic, invisible to the
#          training script, and computed AFTER the VICReg forward pass — meaning
#          the two forward passes saw different dropout states. This has been
#          moved entirely to the training script, which now has full, explicit
#          control over when visual features are zeroed.
#
#  [FIX 2] FiLM activation changed ReLU → GELU for smoother gradient flow.
#          The output gamma is bounded by tanh() to keep scaling in [-1, +1],
#          preventing the FiLM from growing unbounded or collapsing.
#          The final MLP layer is initialized with std=0.02 (non-zero) so that
#          the EEG modulation is numerically non-trivial from the very first step.
#
#  [FIX 3] EEG encoder deepened:
#          - Added a second temporal conv stage (captures alpha/beta bands better)
#          - Transformer widened: dim_feedforward 256→512, layers 2→3
#          - Final projection is now a 2-layer MLP with GELU+Dropout
#          - LayerNorm applied before the transformer for stable training
#
#  [FIX 4] Spatial dropout on visual features reduced: 0.5 → 0.2.
#          The 0.5 rate was aggressively destroying visual features, making the
#          visual branch unreliable and paradoxically reducing pressure on EEG.
#
#  [FIX 5] EEG embedding dropout (p=0.15) added before FiLM conditioning.
#          This regularizes the EEG conditioning path and prevents overfitting
#          to noise artifacts in specific EEG channels.
#
#  [FIX 6] get_eeg_features() method exposed for use by the training script
#          to compute the EEG Discrimination Loss and VICReg without a second
#          duplicate forward pass through the full model.
#
#  [FIX 7] apply_spatial_dropout parameter added to forward() so the training
#          script can disable visual dropout during the EEG-only auxiliary pass.
# =============================================================================


class EEGAttentionEncoder(nn.Module):
    """
    Enhanced spatial-temporal EEG encoder.

    Architecture:
        (B, 32, 250)
          → Conv1d stage 1  [temporal: coarse oscillations]  → (B, 64, 125)
          → Conv1d stage 2  [temporal: fine oscillations]    → (B, 64, 125)
          → Conv1d          [spatial: channel mixing]        → (B, 128, 125)
          → AdaptiveAvgPool1d(64)                            → (B, 128,  64)
          → LayerNorm → TransformerEncoder(layers=3)         → (B,  64, 128)
          → Flatten → MLP(128*64 → 1024 → embed_dim)        → (B, embed_dim)
    """
    def __init__(self, channels=32, time_pts=250, embed_dim=512):
        super().__init__()

        # Stage 1: coarse temporal features (delta / theta oscillations)
        self.temp_conv1 = nn.Conv1d(channels, 64, kernel_size=15, stride=2, padding=7)
        self.bn_temp1   = nn.BatchNorm1d(64)

        # Stage 2: fine temporal features (alpha / beta oscillations)
        self.temp_conv2 = nn.Conv1d(64, 64, kernel_size=7, stride=1, padding=3)
        self.bn_temp2   = nn.BatchNorm1d(64)

        # Spatial mixing: blend inter-channel relationships
        self.spat_conv = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn_spat   = nn.BatchNorm1d(128)

        self.gelu = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(64)  # fixed-size temporal context window

        # LayerNorm before transformer stabilises gradient magnitude
        self.pre_norm = nn.LayerNorm(128)

        # Transformer: captures long-range temporal dependencies across the 64-step context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=512,   # wider than v1 (256) for richer representations
            activation='gelu',
            dropout=0.1,
            batch_first=True,
            norm_first=True        # Pre-LN for stable deep transformer training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3,
                                                   enable_nested_tensor=False)  # 3 vs 2 in v1

        # 2-layer projection MLP  (128*64=8192 → 1024 → embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(128 * 64, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, embed_dim),
        )

    def forward(self, x):
        # x: (B, 32, 250)
        x = self.gelu(self.bn_temp1(self.temp_conv1(x)))   # (B, 64, 125)
        x = self.gelu(self.bn_temp2(self.temp_conv2(x)))   # (B, 64, 125)
        x = self.gelu(self.bn_spat(self.spat_conv(x)))     # (B, 128, 125)
        x = self.pool(x)                                    # (B, 128, 64)

        x = x.transpose(1, 2)                              # (B, 64, 128) — (seq, d_model)
        x = self.pre_norm(x)
        x = self.transformer(x)                            # (B, 64, 128)

        x = x.reshape(x.size(0), -1)                      # (B, 8192)
        x = self.fc(x)                                     # (B, embed_dim)
        return x


class FiLMLayer(nn.Module):
    """
    Enhanced Feature-wise Linear Modulation (FiLM) layer.

    Changes vs. v1:
      - ReLU → GELU for smoother gradients throughout the MLP
      - gamma is bounded by tanh() so scaling stays in [-1, +1]
        (prevents unbounded feature amplification / suppression)
      - Wider intermediate dimension (cond_dim, not cond_dim//2)
      - Final layer initialised with std=0.02 so initial modulation is
        numerically non-trivial and receives immediate gradient signal

    FiLM formula:  FiLM(x) = x · (1 + tanh(γ(z)))  +  β(z)
    """
    def __init__(self, feature_channels, cond_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cond_dim, 2 * feature_channels),
        )
        # Non-trivial initialisation so modulation is non-zero from step 1
        nn.init.normal_(self.mlp[-1].weight, std=0.02)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, cond):
        """
        x    : (B, C, H, W) — visual feature map
        cond : (B, cond_dim) — EEG conditioning vector
        """
        params = self.mlp(cond).unsqueeze(-1).unsqueeze(-1)  # (B, 2C, 1, 1)
        gamma, beta = torch.chunk(params, 2, dim=1)
        gamma = torch.tanh(gamma)                             # bound to [-1, +1]
        return x * (1.0 + gamma) + beta


class BrainGazeDiffusionModel(nn.Module):
    """
    BrainGaze-Diffusion v2  —  EEG-guided Visual Saliency Prediction.

    Architecture overview:
        EEG signal  →  EEGAttentionEncoder  →  eeg_feat (B, 512)
        Image       →  Frozen ResNet-18     →  feat1/2/3 (multi-scale)
        feat3 + eeg_feat  →  FiLM1 → UpConv → SkipCat(feat2) →
        feat2' + eeg_feat →  FiLM2 → UpConv → SkipCat(feat1) →
        feat1' + eeg_feat →  FiLM3 → UpConv → UpConv → Conv → Sigmoid
                                                    ↓
                                       Saliency map (B, 1, 224, 224)

    Key design differences from v1:
      - No random modality dropout inside forward() — controlled externally
      - get_eeg_features() exposed for discrimination / VICReg losses
      - apply_spatial_dropout arg gives training script fine-grained control
      - Spatial dropout reduced 0.5 → 0.2
      - EEG embedding dropout (p=0.15) regularizes the conditioning path
    """

    def __init__(self, embed_dim=512):
        super().__init__()

        # ── 1. EEG encoder ──────────────────────────────────────────────────
        self.eeg_encoder = EEGAttentionEncoder(embed_dim=embed_dim)
        # Regularise EEG conditioning vector (not individual channels)
        self.eeg_dropout = nn.Dropout(p=0.15)

        # ── 2. Visual prior encoder (Frozen ResNet-18) ───────────────────────
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.early_layers = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )                           # → (B, 64, 56, 56)
        self.layer1 = resnet.layer1 # → (B,  64, 56, 56)  [skip 2]
        self.layer2 = resnet.layer2 # → (B, 128, 28, 28)  [skip 1]
        self.layer3 = resnet.layer3 # → (B, 256, 14, 14)  [bottleneck]

        for layer in [self.early_layers, self.layer1, self.layer2, self.layer3]:
            for param in layer.parameters():
                param.requires_grad = False

        self.spatial_dropout = nn.Dropout2d(p=0.2)   # reduced from 0.5

        # ── 3. Subject embedding ─────────────────────────────────────────────
        self.subject_embed = nn.Embedding(20, embed_dim)
        nn.init.normal_(self.subject_embed.weight, std=0.01)

        # ── 4. FiLM modulation blocks ────────────────────────────────────────
        self.film1 = FiLMLayer(256, cond_dim=embed_dim)
        self.film2 = FiLMLayer(128, cond_dim=embed_dim)
        self.film3 = FiLMLayer(64,  cond_dim=embed_dim)

        # ── 5. U-Net skip decoder ────────────────────────────────────────────
        # Block 1:  14×14 → 28×28
        self.upconv1    = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv_skip1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)   # 128+128→128
        self.bn_up1     = nn.BatchNorm2d(128)

        # Block 2:  28×28 → 56×56
        self.upconv2    = nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1)
        self.conv_skip2 = nn.Conv2d(128, 64,  kernel_size=3, padding=1)   # 64+64→64
        self.bn_up2     = nn.BatchNorm2d(64)

        # Block 3:  56×56 → 112×112
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn_up3  = nn.BatchNorm2d(32)

        # Block 4:  112×112 → 224×224
        self.final_up = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.bn_final = nn.BatchNorm2d(16)

        self.conv_out = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    # ─────────────────────────────────────────────────────────────────────────
    def get_eeg_features(self, eeg, subject_ids=None):
        """
        Returns the raw EEG embedding before dropout.
        Used by the training script for:
          • VICReg variance loss  (keeps embeddings spread out)
          • EEG discrimination loss  (real vs. shuffled EEG)
        Calling this avoids a redundant full forward pass through the decoder.
        """
        feat = self.eeg_encoder(eeg)
        if subject_ids is not None:
            feat = feat + self.subject_embed(subject_ids)
        return feat

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, eeg, image, subject_ids=None,
                zero_image=False, apply_spatial_dropout=True):
        """
        Forward pass.

        Args:
            zero_image            : Zero all visual encoder outputs (EEG-only pass).
            apply_spatial_dropout : Apply Dropout2d on feat3 (set False for clean eval
                                    passes and the EEG-only auxiliary loss).
        """
        # ── EEG conditioning vector ──────────────────────────────────────────
        eeg_feat = self.eeg_encoder(eeg)                          # (B, 512)
        if subject_ids is not None:
            eeg_feat = eeg_feat + self.subject_embed(subject_ids)
        if self.training:
            eeg_feat = self.eeg_dropout(eeg_feat)

        # ── Visual features (frozen, no grad) ───────────────────────────────
        with torch.no_grad():
            feat0 = self.early_layers(image)
            feat1 = self.layer1(feat0)   # (B,  64, 56, 56)
            feat2 = self.layer2(feat1)   # (B, 128, 28, 28)
            feat3 = self.layer3(feat2)   # (B, 256, 14, 14)

        if zero_image:
            feat1 = torch.zeros_like(feat1)
            feat2 = torch.zeros_like(feat2)
            feat3 = torch.zeros_like(feat3)
        elif self.training and apply_spatial_dropout:
            feat3 = self.spatial_dropout(feat3)

        # ── Decoding with FiLM-modulated skip connections ───────────────────
        # Block 1 — 14×14 → 28×28
        x = self.film1(feat3, eeg_feat)
        x = self.upconv1(x)
        x = torch.cat([x, feat2], dim=1)          # skip connection
        x = self.relu(self.bn_up1(self.conv_skip1(x)))

        # Block 2 — 28×28 → 56×56
        x = self.film2(x, eeg_feat)
        x = self.upconv2(x)
        x = torch.cat([x, feat1], dim=1)          # skip connection
        x = self.relu(self.bn_up2(self.conv_skip2(x)))

        # Block 3 — 56×56 → 112×112
        x = self.film3(x, eeg_feat)
        x = self.relu(self.bn_up3(self.upconv3(x)))

        # Final — 112×112 → 224×224
        x = self.relu(self.bn_final(self.final_up(x)))
        return self.sigmoid(self.conv_out(x))


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    eeg         = torch.randn(4, 32, 250)
    image       = torch.randn(4, 3, 224, 224)
    subject_ids = torch.randint(0, 20, (4,))

    model = BrainGazeDiffusionModel()
    model.eval()

    out          = model(eeg, image, subject_ids)
    eeg_feats    = model.get_eeg_features(eeg, subject_ids)
    out_eeg_only = model(eeg, image, subject_ids, zero_image=True)

    print(f"Saliency output shape   : {out.shape}")           # (4, 1, 224, 224)
    print(f"EEG features shape      : {eeg_feats.shape}")     # (4, 512)
    print(f"EEG-only output shape   : {out_eeg_only.shape}")  # (4, 1, 224, 224)

    total_params   = sum(p.numel() for p in model.parameters())
    trainable      = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters        : {total_params:,}")
    print(f"Trainable parameters    : {trainable:,}")
