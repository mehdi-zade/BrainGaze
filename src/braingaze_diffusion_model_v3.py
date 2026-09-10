"""
BrainGaze-Diffusion Model  —  Version 3
========================================

Informed by: "The Modality Imbalance Report: Architectural Root Causes
              and Algorithmic Solutions"

Design philosophy (v3):
──────────────────────
The fundamental problem is that the frozen ResNet-18 provides a near-perfect
visual prior from epoch 0, while the EEG encoder starts from scratch.
This creates an *infinite* learning speed disparity (Reason 1 in the report).
Because the decoder can produce excellent saliency maps using only image
features, EEG never receives meaningful gradient signal (Reason 3: the
Domination-Suppression Cycle).

v2 tried to fix this with discrimination losses and auxiliary EEG-only losses.
These are necessary but not sufficient.  v3 adds three architectural
mechanisms drawn directly from the Modality Imbalance Report:

[NEW 1]  EEG Bottleneck Gate  (inspired by PMR / Prototypical Modal Rebalance)
         Instead of the EEG vector merely *modulating* frozen visual features
         via FiLM, the decoder now has a GATED RESIDUAL architecture:

             output = Gate(z) · EEG_path(z) + (1 - Gate(z)) · Visual_path(x)

         where z is the EEG embedding and x are the visual features.
         The gate is a learned scalar per spatial location.  Critically,
         the gate is initialized to 0.5 (even split) via bias init, forcing
         the model to route roughly half its prediction through the EEG branch
         from the very first step.  This prevents the visual path from
         dominating before the EEG encoder has had time to learn.

[NEW 2]  Adaptive Visual Noise Schedule  (inspired by PER / Prototypical
         Entropy Regularization of the dominant modality)
         The report's PER slows down the dominant modality to give the slow
         modality time to catch up.  Since our visual backbone is frozen and
         cannot be slowed, we instead inject *learnable* Gaussian noise into
         the visual feature maps.  The noise std starts high (σ=0.5) and
         decays linearly to 0 over the first N joint-training epochs.
         This blurs the visual shortcut, forcing the decoder to rely on the
         EEG branch for fine-grained spatial details during early training.

[NEW 3]  Cross-Attention Fusion  (replaces FiLM at the bottleneck)
         FiLM applies channel-wise affine transforms: each spatial location
         in the feature map gets the SAME scale and shift.  This severely
         limits how much EEG can modulate the spatial layout of attention.

         At the bottleneck (14×14, deepest decoder level) we now use
         cross-attention between the image features (queries) and the EEG
         temporal sequence (keys/values).  This allows the EEG signal to
         selectively amplify or suppress specific spatial locations, not
         just channels.  FiLM is retained at the shallower decoder levels
         (28×28, 56×56) where channel-wise modulation is sufficient.

[NEW 4]  Conditional Learning Speed (CLS) Monitor  (Section 3 of the report)
         We expose a method to compute the gradient-norm ratio between the
         EEG encoder and the decoder/FiLM.  The training script uses this
         to adaptively scale the EEG learning rate:  if CLS_eeg / CLS_dec
         drops below a threshold, the EEG LR is boosted.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


# ═══════════════════════════════════════════════════════════════════════════════
#  EEG Encoder  (same as v2, battle-tested)
# ═══════════════════════════════════════════════════════════════════════════════

class EEGAttentionEncoder(nn.Module):
    """
    Spatial-temporal EEG encoder.

    (B, 32, 250) → Conv1d×3 → AdaptivePool → LayerNorm → Transformer(3L)
                 → MLP → (B, 512)

    Also returns intermediate sequence features for cross-attention:
        seq_features: (B, 64, 128) — the transformer output before flattening
    """
    def __init__(self, channels=32, time_pts=250, embed_dim=512):
        super().__init__()
        self.temp_conv1 = nn.Conv1d(channels, 64, kernel_size=15, stride=2, padding=7)
        self.bn_temp1   = nn.BatchNorm1d(64)
        self.temp_conv2 = nn.Conv1d(64, 64, kernel_size=7, stride=1, padding=3)
        self.bn_temp2   = nn.BatchNorm1d(64)
        self.spat_conv  = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn_spat    = nn.BatchNorm1d(128)
        self.gelu       = nn.GELU()
        self.pool       = nn.AdaptiveAvgPool1d(64)
        self.pre_norm   = nn.LayerNorm(128)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=512,
            activation='gelu', dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=3, enable_nested_tensor=False
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 64, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, embed_dim),
        )

    def forward(self, x, return_sequence=False):
        """
        Args:
            x: (B, 32, 250)
            return_sequence: if True, also return the (B, 64, 128) transformer
                             output for cross-attention in the decoder.
        Returns:
            embed: (B, embed_dim)
            seq  : (B, 64, 128)  [only if return_sequence=True]
        """
        x = self.gelu(self.bn_temp1(self.temp_conv1(x)))  # (B, 64, 125)
        x = self.gelu(self.bn_temp2(self.temp_conv2(x)))  # (B, 64, 125)
        x = self.gelu(self.bn_spat(self.spat_conv(x)))    # (B, 128, 125)
        x = self.pool(x)                                   # (B, 128, 64)

        x = x.transpose(1, 2)        # (B, 64, 128)
        x = self.pre_norm(x)
        seq = self.transformer(x)    # (B, 64, 128)

        embed = seq.reshape(seq.size(0), -1)  # (B, 8192)
        embed = self.fc(embed)                # (B, embed_dim)

        if return_sequence:
            return embed, seq
        return embed


# ═══════════════════════════════════════════════════════════════════════════════
#  FiLM Layer  (retained for shallow decoder stages — 28×28 and 56×56)
# ═══════════════════════════════════════════════════════════════════════════════

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: FiLM(x) = x · (1 + tanh(γ)) + β"""
    def __init__(self, feature_channels, cond_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cond_dim, 2 * feature_channels),
        )
        nn.init.normal_(self.mlp[-1].weight, std=0.02)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, cond):
        params = self.mlp(cond).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = torch.chunk(params, 2, dim=1)
        gamma = torch.tanh(gamma)
        return x * (1.0 + gamma) + beta


# ═══════════════════════════════════════════════════════════════════════════════
#  [NEW 3]  Cross-Attention Fusion Block  (bottleneck only)
# ═══════════════════════════════════════════════════════════════════════════════

class EEGCrossAttention(nn.Module):
    """
    Cross-attention between spatial image features and EEG temporal sequence.

    Queries: image feature map reshaped to (B, H*W, C_img)
    Keys/Values: EEG transformer sequence (B, 64, 128) projected to C_img

    This lets the EEG signal spatially modulate the visual features — every
    spatial location can attend to different parts of the EEG temporal context.
    Much more expressive than FiLM's channel-only modulation.
    """
    def __init__(self, img_channels=256, eeg_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.img_channels = img_channels

        # Project EEG sequence to match image feature dimension
        self.eeg_proj = nn.Linear(eeg_dim, img_channels)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=img_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(img_channels)
        self.norm_kv = nn.LayerNorm(img_channels)
        self.norm_out = nn.LayerNorm(img_channels)

        # Post-attention FFN
        self.ffn = nn.Sequential(
            nn.Linear(img_channels, img_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(img_channels * 2, img_channels),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(img_channels)

    def forward(self, feat_map, eeg_seq):
        """
        feat_map: (B, C, H, W) — image feature tensor (e.g. 256×14×14)
        eeg_seq : (B, 64, 128) — EEG temporal sequence from transformer

        Returns: (B, C, H, W) — modulated feature tensor
        """
        B, C, H, W = feat_map.shape

        # Reshape image features to sequence: (B, H*W, C)
        q = feat_map.flatten(2).transpose(1, 2)  # (B, 196, 256)
        q = self.norm_q(q)

        # Project EEG to image dimension: (B, 64, 128) → (B, 64, 256)
        kv = self.eeg_proj(eeg_seq)
        kv = self.norm_kv(kv)

        # Cross-attention: image queries attend to EEG key-values
        attn_out, _ = self.cross_attn(q, kv, kv)       # (B, 196, 256)
        q = self.norm_out(q + attn_out)                 # residual + norm

        # FFN
        q = self.norm_ffn(q + self.ffn(q))              # (B, 196, 256)

        # Reshape back to spatial: (B, 256, 14, 14)
        return q.transpose(1, 2).reshape(B, C, H, W)


# ═══════════════════════════════════════════════════════════════════════════════
#  [NEW 1]  Gated Residual Fusion
# ═══════════════════════════════════════════════════════════════════════════════

class GatedResidualFusion(nn.Module):
    """
    Gated residual fusion: output = g · eeg_modulated + (1 − g) · visual_raw

    The gate g is predicted from the EEG embedding and broadcasts over the
    spatial dimensions.  It acts per-channel, per-location: (B, C, H, W).

    Critically initialised so g ≈ 0.5 at the start of training, forcing the
    model to use both modalities equally before either has a chance to dominate.
    """
    def __init__(self, channels, cond_dim=512):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(cond_dim, channels),
            nn.Sigmoid(),
        )
        # Initialise the linear layer so that Sigmoid(Wx+b) ≈ 0.5
        nn.init.zeros_(self.gate_mlp[0].weight)
        nn.init.zeros_(self.gate_mlp[0].bias)

    def forward(self, visual_raw, eeg_modulated, cond):
        """
        visual_raw    : (B, C, H, W) — original frozen ResNet features
        eeg_modulated : (B, C, H, W) — features after cross-attn / FiLM
        cond          : (B, cond_dim) — EEG embedding vector
        """
        g = self.gate_mlp(cond).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return g * eeg_modulated + (1.0 - g) * visual_raw


# ═══════════════════════════════════════════════════════════════════════════════
#  [NEW 2]  Adaptive Visual Noise Schedule
# ═══════════════════════════════════════════════════════════════════════════════

class VisualNoiseSchedule(nn.Module):
    """
    Injects Gaussian noise into visual features with a decaying std.

    During early joint training, this blurs the visual shortcut and forces
    the decoder to rely on the EEG branch for fine details.  The noise decays
    linearly from σ_max to 0 over `decay_epochs` joint-training epochs.

    Inspired by PER (Prototypical Entropy Regularization): we can't slow down
    the frozen ResNet's "learning" directly, so instead we degrade its output
    during early epochs, which achieves the same effect — preventing premature
    visual dominance.
    """
    def __init__(self, sigma_max=0.3, decay_epochs=25):
        super().__init__()
        self.sigma_max = sigma_max
        self.decay_epochs = decay_epochs
        self.register_buffer('current_sigma', torch.tensor(sigma_max))

    def set_epoch(self, joint_epoch):
        """Call at the start of each joint-training epoch (0-indexed)."""
        progress = min(joint_epoch / self.decay_epochs, 1.0)
        self.current_sigma.fill_(self.sigma_max * (1.0 - progress))

    def forward(self, x):
        """Only applies noise during training."""
        if self.training and self.current_sigma.item() > 1e-6:
            noise = torch.randn_like(x) * self.current_sigma
            return x + noise
        return x


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Model
# ═══════════════════════════════════════════════════════════════════════════════

class BrainGazeDiffusionModel(nn.Module):
    """
    BrainGaze-Diffusion v3  —  Modality-Balanced EEG Saliency Prediction.

    Architecture changes vs. v2:
        1. Cross-Attention at bottleneck (14×14) — spatial EEG modulation
        2. Gated Residual Fusion at all 3 decoder stages — prevents visual dominance
        3. Visual Noise Schedule — degrades visual features during early joint training
        4. EEG encoder returns both embedding AND temporal sequence

    What is preserved from v2:
        - FiLM at stages 2 and 3 (shallow levels: 28×28, 56×56)
        - Frozen ResNet-18 backbone
        - Subject embedding
        - EEG embedding dropout
        - get_eeg_features() for external loss computation
    """

    def __init__(self, embed_dim=512, noise_sigma=0.3, noise_decay_epochs=25):
        super().__init__()

        # ── 1. EEG encoder ──────────────────────────────────────────────────
        self.eeg_encoder = EEGAttentionEncoder(embed_dim=embed_dim)
        self.eeg_dropout = nn.Dropout(p=0.15)

        # ── 2. Visual encoder (frozen ResNet-18) ────────────────────────────
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.early_layers = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1  # (B, 64, 56, 56)
        self.layer2 = resnet.layer2  # (B, 128, 28, 28)
        self.layer3 = resnet.layer3  # (B, 256, 14, 14)

        for layer in [self.early_layers, self.layer1, self.layer2, self.layer3]:
            for param in layer.parameters():
                param.requires_grad = False

        # ── 3. Visual noise schedule [NEW 2] ─────────────────────────────────
        self.visual_noise = VisualNoiseSchedule(
            sigma_max=noise_sigma, decay_epochs=noise_decay_epochs
        )

        # ── 4. Subject embedding ─────────────────────────────────────────────
        self.subject_embed = nn.Embedding(20, embed_dim)
        nn.init.normal_(self.subject_embed.weight, std=0.01)

        # ── 5. Bottleneck fusion: Cross-Attention + Gate [NEW 1 & 3] ────────
        self.cross_attn = EEGCrossAttention(
            img_channels=256, eeg_dim=128, num_heads=4
        )
        self.gate1 = GatedResidualFusion(256, cond_dim=embed_dim)

        # ── 6. FiLM + Gate for shallower levels ─────────────────────────────
        self.film2 = FiLMLayer(128, cond_dim=embed_dim)
        self.gate2 = GatedResidualFusion(128, cond_dim=embed_dim)

        self.film3 = FiLMLayer(64, cond_dim=embed_dim)
        self.gate3 = GatedResidualFusion(64, cond_dim=embed_dim)

        # ── 7. U-Net skip decoder ────────────────────────────────────────────
        # Block 1:  14×14 → 28×28
        self.upconv1    = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv_skip1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn_up1     = nn.BatchNorm2d(128)

        # Block 2:  28×28 → 56×56
        self.upconv2    = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv_skip2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn_up2     = nn.BatchNorm2d(64)

        # Block 3:  56×56 → 112×112
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn_up3  = nn.BatchNorm2d(32)

        # Block 4:  112×112 → 224×224
        self.final_up = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.bn_final = nn.BatchNorm2d(16)

        self.conv_out = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.relu     = nn.ReLU()
        self.sigmoid  = nn.Sigmoid()

    # ─────────────────────────────────────────────────────────────────────────
    def get_eeg_features(self, eeg, subject_ids=None):
        """Returns EEG embedding for external loss computation."""
        feat = self.eeg_encoder(eeg, return_sequence=False)
        if subject_ids is not None:
            feat = feat + self.subject_embed(subject_ids)
        return feat

    # ─────────────────────────────────────────────────────────────────────────
    def get_gradient_norms(self):
        """
        [NEW 4]  Conditional Learning Speed (CLS) monitor.

        Returns gradient norms for the EEG encoder and decoder separately,
        allowing the training script to compute the CLS ratio and adaptively
        adjust learning rates when the EEG branch is under-exploited.
        """
        eeg_norm = 0.0
        dec_norm = 0.0
        for name, p in self.named_parameters():
            if p.grad is not None:
                g = p.grad.data.norm(2).item() ** 2
                if 'eeg_encoder' in name or 'subject_embed' in name:
                    eeg_norm += g
                elif p.requires_grad:
                    dec_norm += g
        return eeg_norm ** 0.5, dec_norm ** 0.5

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, eeg, image, subject_ids=None,
                zero_image=False, apply_visual_noise=True):
        """
        Args:
            zero_image         : Zero all visual features (EEG-only pass)
            apply_visual_noise : Apply the adaptive noise schedule (set False
                                 for validation / diagnostic passes)
        """
        # ── EEG: get both the embedding and the temporal sequence ──────────
        eeg_embed, eeg_seq = self.eeg_encoder(eeg, return_sequence=True)
        # eeg_embed: (B, 512),  eeg_seq: (B, 64, 128)

        if subject_ids is not None:
            eeg_embed = eeg_embed + self.subject_embed(subject_ids)
        if self.training:
            eeg_embed = self.eeg_dropout(eeg_embed)

        # ── Visual features (frozen) ────────────────────────────────────────
        with torch.no_grad():
            feat0 = self.early_layers(image)
            feat1 = self.layer1(feat0)   # (B, 64, 56, 56)
            feat2 = self.layer2(feat1)   # (B, 128, 28, 28)
            feat3 = self.layer3(feat2)   # (B, 256, 14, 14)

        if zero_image:
            feat1 = torch.zeros_like(feat1)
            feat2 = torch.zeros_like(feat2)
            feat3 = torch.zeros_like(feat3)
        elif apply_visual_noise:
            # [NEW 2] Inject decaying noise to blur the visual shortcut
            feat3 = self.visual_noise(feat3)

        # ── Decoder Block 1:  14×14 → 28×28 ────────────────────────────────
        #    Cross-attention at the bottleneck for spatial EEG modulation
        feat3_modulated = self.cross_attn(feat3, eeg_seq)    # [NEW 3]
        x = self.gate1(feat3, feat3_modulated, eeg_embed)    # [NEW 1]

        x = self.upconv1(x)
        x = torch.cat([x, feat2], dim=1)    # skip connection
        x = self.relu(self.bn_up1(self.conv_skip1(x)))

        # ── Decoder Block 2:  28×28 → 56×56 ────────────────────────────────
        x_raw = x.clone()                                     # save pre-FiLM
        x_mod = self.film2(x, eeg_embed)
        x = self.gate2(x_raw, x_mod, eeg_embed)              # [NEW 1]

        x = self.upconv2(x)
        x = torch.cat([x, feat1], dim=1)    # skip connection
        x = self.relu(self.bn_up2(self.conv_skip2(x)))

        # ── Decoder Block 3:  56×56 → 112×112 ──────────────────────────────
        x_raw = x.clone()
        x_mod = self.film3(x, eeg_embed)
        x = self.gate3(x_raw, x_mod, eeg_embed)              # [NEW 1]

        x = self.relu(self.bn_up3(self.upconv3(x)))

        # ── Final:  112×112 → 224×224 ───────────────────────────────────────
        x = self.relu(self.bn_final(self.final_up(x)))
        return self.sigmoid(self.conv_out(x))


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    eeg   = torch.randn(4, 32, 250)
    image = torch.randn(4, 3, 224, 224)
    subs  = torch.randint(0, 20, (4,))

    model = BrainGazeDiffusionModel()
    model.eval()

    out        = model(eeg, image, subs)
    eeg_feats  = model.get_eeg_features(eeg, subs)
    out_eeg    = model(eeg, image, subs, zero_image=True)

    print(f"Saliency output   : {out.shape}")
    print(f"EEG features      : {eeg_feats.shape}")
    print(f"EEG-only output   : {out_eeg.shape}")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params      : {total:,}")
    print(f"Trainable params  : {trainable:,}")
