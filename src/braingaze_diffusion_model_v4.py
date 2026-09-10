"""
BrainGaze-Diffusion Model  —  Version 4
========================================

Changelog vs v3:
────────────────
[FIX] Guaranteed Minimum Gate Opening (min_gate = 0.25)
      In v3, Gate 2 (middle) and Gate 3 (shallow) collapsed to ~0.001 and ~0.010,
      meaning the model successfully bypassed EEG at those levels.
      v4 solves this by enforcing a minimum gate opening of 0.25:
          g = min_gate + (1.0 - min_gate) * Sigmoid(gate_mlp(cond))
      At initialization, the gate is at 0.625 (62.5% EEG).
      Even if the model tries to ignore EEG, it is mathematically forced to route
      at least 25% of features through the EEG-modulated branch. This removes the
      unimodality shortcut and forces the network to learn helpful EEG features.
"""

import torch
import torch.nn as nn
import torchvision.models as models

# ═══════════════════════════════════════════════════════════════════════════════
#  EEG Encoder
# ═══════════════════════════════════════════════════════════════════════════════

class EEGAttentionEncoder(nn.Module):
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
        x = self.gelu(self.bn_temp1(self.temp_conv1(x)))
        x = self.gelu(self.bn_temp2(self.temp_conv2(x)))
        x = self.gelu(self.bn_spat(self.spat_conv(x)))
        x = self.pool(x)

        x = x.transpose(1, 2)
        x = self.pre_norm(x)
        seq = self.transformer(x)

        embed = seq.reshape(seq.size(0), -1)
        embed = self.fc(embed)

        if return_sequence:
            return embed, seq
        return embed


# ═══════════════════════════════════════════════════════════════════════════════
#  FiLM Layer
# ═══════════════════════════════════════════════════════════════════════════════

class FiLMLayer(nn.Module):
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
#  Cross-Attention Fusion Block
# ═══════════════════════════════════════════════════════════════════════════════

class EEGCrossAttention(nn.Module):
    def __init__(self, img_channels=256, eeg_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.img_channels = img_channels
        self.eeg_proj = nn.Linear(eeg_dim, img_channels)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=img_channels, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(img_channels)
        self.norm_kv = nn.LayerNorm(img_channels)
        self.norm_out = nn.LayerNorm(img_channels)

        self.ffn = nn.Sequential(
            nn.Linear(img_channels, img_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(img_channels * 2, img_channels),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(img_channels)

    def forward(self, feat_map, eeg_seq):
        B, C, H, W = feat_map.shape
        q = feat_map.flatten(2).transpose(1, 2)
        q = self.norm_q(q)

        kv = self.eeg_proj(eeg_seq)
        kv = self.norm_kv(kv)

        attn_out, _ = self.cross_attn(q, kv, kv)
        q = self.norm_out(q + attn_out)
        q = self.norm_ffn(q + self.ffn(q))

        return q.transpose(1, 2).reshape(B, C, H, W)


# ═══════════════════════════════════════════════════════════════════════════════
#  [FIX] Gated Residual Fusion with Minimum Gate
# ═══════════════════════════════════════════════════════════════════════════════

class GatedResidualFusion(nn.Module):
    def __init__(self, channels, cond_dim=512, min_gate=0.25):
        super().__init__()
        self.min_gate = min_gate
        self.gate_mlp = nn.Sequential(
            nn.Linear(cond_dim, channels),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.gate_mlp[0].weight)
        nn.init.zeros_(self.gate_mlp[0].bias)

    def forward(self, visual_raw, eeg_modulated, cond):
        g_raw = self.gate_mlp(cond).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        # Force gate to be in range [min_gate, 1.0]
        g = self.min_gate + (1.0 - self.min_gate) * g_raw
        return g * eeg_modulated + (1.0 - g) * visual_raw


# ═══════════════════════════════════════════════════════════════════════════════
#  Adaptive Visual Noise Schedule
# ═══════════════════════════════════════════════════════════════════════════════

class VisualNoiseSchedule(nn.Module):
    def __init__(self, sigma_max=0.3, decay_epochs=25):
        super().__init__()
        self.sigma_max = sigma_max
        self.decay_epochs = decay_epochs
        self.register_buffer('current_sigma', torch.tensor(sigma_max))

    def set_epoch(self, joint_epoch):
        progress = min(joint_epoch / self.decay_epochs, 1.0)
        self.current_sigma.fill_(self.sigma_max * (1.0 - progress))

    def forward(self, x):
        if self.training and self.current_sigma.item() > 1e-6:
            noise = torch.randn_like(x) * self.current_sigma
            return x + noise
        return x


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Model v4
# ═══════════════════════════════════════════════════════════════════════════════

class BrainGazeDiffusionModel(nn.Module):
    def __init__(self, embed_dim=512, noise_sigma=0.3, noise_decay_epochs=25, min_gate=0.25):
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

        # ── 3. Visual noise schedule ─────────────────────────────────────────
        self.visual_noise = VisualNoiseSchedule(
            sigma_max=noise_sigma, decay_epochs=noise_decay_epochs
        )

        # ── 4. Subject embedding ─────────────────────────────────────────────
        self.subject_embed = nn.Embedding(20, embed_dim)
        nn.init.normal_(self.subject_embed.weight, std=0.01)

        # ── 5. Bottleneck fusion: Cross-Attention + Gate ────────────────────
        self.cross_attn = EEGCrossAttention(img_channels=256, eeg_dim=128, num_heads=4)
        self.gate1 = GatedResidualFusion(256, cond_dim=embed_dim, min_gate=min_gate)

        # ── 6. FiLM + Gate ──────────────────────────────────────────────────
        self.film2 = FiLMLayer(128, cond_dim=embed_dim)
        self.gate2 = GatedResidualFusion(128, cond_dim=embed_dim, min_gate=min_gate)

        self.film3 = FiLMLayer(64, cond_dim=embed_dim)
        self.gate3 = GatedResidualFusion(64, cond_dim=embed_dim, min_gate=min_gate)

        # ── 7. U-Net skip decoder ────────────────────────────────────────────
        self.upconv1    = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv_skip1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn_up1     = nn.BatchNorm2d(128)

        self.upconv2    = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv_skip2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn_up2     = nn.BatchNorm2d(64)

        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn_up3  = nn.BatchNorm2d(32)

        self.final_up = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.bn_final = nn.BatchNorm2d(16)

        self.conv_out = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.relu     = nn.ReLU()
        self.sigmoid  = nn.Sigmoid()

    def get_eeg_features(self, eeg, subject_ids=None):
        feat = self.eeg_encoder(eeg, return_sequence=False)
        if subject_ids is not None:
            feat = feat + self.subject_embed(subject_ids)
        return feat

    def get_gradient_norms(self):
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

    def forward(self, eeg, image, subject_ids=None, zero_image=False, apply_visual_noise=True):
        eeg_embed, eeg_seq = self.eeg_encoder(eeg, return_sequence=True)

        if subject_ids is not None:
            eeg_embed = eeg_embed + self.subject_embed(subject_ids)
        if self.training:
            eeg_embed = self.eeg_dropout(eeg_embed)

        with torch.no_grad():
            feat0 = self.early_layers(image)
            feat1 = self.layer1(feat0)
            feat2 = self.layer2(feat1)
            feat3 = self.layer3(feat2)

        if zero_image:
            feat1 = torch.zeros_like(feat1)
            feat2 = torch.zeros_like(feat2)
            feat3 = torch.zeros_like(feat3)
        elif apply_visual_noise:
            feat3 = self.visual_noise(feat3)

        # Decoder Block 1: 14x14 -> 28x28
        feat3_modulated = self.cross_attn(feat3, eeg_seq)
        x = self.gate1(feat3, feat3_modulated, eeg_embed)

        x = self.upconv1(x)
        x = torch.cat([x, feat2], dim=1)
        x = self.relu(self.bn_up1(self.conv_skip1(x)))

        # Decoder Block 2: 28x28 -> 56x56
        x_raw = x.clone()
        x_mod = self.film2(x, eeg_embed)
        x = self.gate2(x_raw, x_mod, eeg_embed)

        x = self.upconv2(x)
        x = torch.cat([x, feat1], dim=1)
        x = self.relu(self.bn_up2(self.conv_skip2(x)))

        # Decoder Block 3: 56x56 -> 112x112
        x_raw = x.clone()
        x_mod = self.film3(x, eeg_embed)
        x = self.gate3(x_raw, x_mod, eeg_embed)

        x = self.relu(self.bn_up3(self.upconv3(x)))

        # Final
        x = self.relu(self.bn_final(self.final_up(x)))
        return self.sigmoid(self.conv_out(x))
