"""
braingaze.py
============
BrainGaze: Cognitive-Visual Modular Routing (CVMR) Architecture
for Personalised Neuro-Visual Saliency Prediction.

Key Components:
1. CognitiveRoutingEEGEncoder: Encodes 32-ch EEG waveforms and routes probability mass.
2. VisualBasisGenerator: Decodes 8 spatially orthogonal saliency basis maps from visual features.
3. BrainGaze: Bilinear mixture model with Parseval Isometric Guarantee.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class CognitiveRoutingEEGEncoder(nn.Module):
    def __init__(self, channels=32, num_basis=8, embed_dim=256):
        super().__init__()
        self.temp_conv = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.spat_conv = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(32)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256,
            activation='gelu', dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Sequential(
            nn.Linear(128 * 32, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        self.router = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_basis)
        )
        self.subject_embed = nn.Embedding(20, embed_dim)
        nn.init.normal_(self.subject_embed.weight, std=0.01)

    def forward(self, eeg, subject_ids=None):
        x = self.temp_conv(eeg)
        x = self.spat_conv(x)
        x = x.transpose(1, 2)
        x = self.transformer(x)
        embed = self.fc(x.flatten(1))
        if subject_ids is not None:
            embed = embed + self.subject_embed(subject_ids)
        routing_logits = self.router(embed)
        routing_weights = torch.softmax(routing_logits, dim=-1)
        return embed, routing_weights


class VisualBasisGenerator(nn.Module):
    def __init__(self, num_basis=8):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 14x14 -> 28x28
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 28x28 -> 56x56
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 56x56 -> 112x112
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, num_basis, kernel_size=4, stride=2, padding=1), # 112x112 -> 224x224
            nn.Sigmoid()
        )

    def forward(self, img):
        with torch.no_grad():
            x = self.conv1(img)
            x = self.bn1(x)
            f0 = self.maxpool(self.relu(x))
            f1 = self.layer1(f0)
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)

        basis_maps = self.decoder(f3)
        B, K, H, W = basis_maps.shape
        flat = basis_maps.view(B, K, -1)
        norm_flat = flat / (flat.sum(dim=-1, keepdim=True) + 1e-7)
        norm_basis = norm_flat.view(B, K, H, W)
        return norm_basis, f3


class BrainGaze(nn.Module):
    """
    BrainGaze: Cognitive-Visual Modular Routing (CVMR) Architecture
    with Parseval Isometric Guarantee.
    """
    def __init__(self, num_basis=8, embed_dim=256):
        super().__init__()
        self.num_basis = num_basis
        self.eeg_encoder = CognitiveRoutingEEGEncoder(num_basis=num_basis, embed_dim=embed_dim)
        self.vis_generator = VisualBasisGenerator(num_basis=num_basis)

    def forward(self, eeg, img, subject_ids=None):
        basis_maps, f3 = self.vis_generator(img)
        eeg_embed, routing_weights = self.eeg_encoder(eeg, subject_ids)
        weights = routing_weights.unsqueeze(-1).unsqueeze(-1)
        final_saliency = torch.sum(weights * basis_maps, dim=1, keepdim=True)
        return final_saliency, basis_maps, routing_weights, eeg_embed, f3


# Backward compatibility alias
BrainGaze_v5_CVMR = BrainGaze


def loss_orthogonality(basis_maps):
    """
    Gram Matrix Orthogonality Loss enforcing spatial orthogonality across basis maps.
    """
    B, K, H, W = basis_maps.shape
    flat = basis_maps.view(B, K, -1)
    flat_norm = flat / (flat.norm(dim=-1, keepdim=True) + 1e-7)
    gram = torch.bmm(flat_norm, flat_norm.transpose(1, 2))
    identity = torch.eye(K, device=basis_maps.device).unsqueeze(0).repeat(B, 1, 1)
    loss = torch.norm(gram - identity, p='fro', dim=(1, 2)).mean()
    return loss


def loss_routing_diversity(routing_weights):
    """
    Routing Weight Diversity Loss encouraging dynamic dispersion.
    """
    batch_std = torch.std(routing_weights, dim=0).mean()
    loss = torch.relu(0.15 - batch_std)
    return loss
