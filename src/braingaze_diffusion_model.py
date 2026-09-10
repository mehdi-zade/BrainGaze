import torch
import torch.nn as nn
import torchvision.models as models

class EEGAttentionEncoder(nn.Module):
    """
    Compact spatial-temporal encoder for preprocessed EEG trials (32 channels, 250 timepoints).
    Uses 1D convolutions to extract temporal/frequency features followed by a transformer block.
    """
    def __init__(self, channels=32, time_pts=250, embed_dim=512):
        super().__init__()
        # Temporal convolution to capture oscillations (delta, theta, alpha, beta)
        self.temp_conv = nn.Conv1d(channels, 64, kernel_size=15, stride=2, padding=7)
        self.bn_temp = nn.BatchNorm1d(64)
        
        # Spatial convolution to blend channel relationships
        self.spat_conv = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn_spat = nn.BatchNorm1d(128)
        
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(64) # Downsample temporal dimension to a fixed size
        
        # Transformer Encoder layer to capture long-range temporal dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, 
            nhead=4, 
            dim_feedforward=256, 
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Final projection to latent attention space
        self.fc = nn.Linear(128 * 64, embed_dim)
        
    def forward(self, x):
        # x shape: (B, 32, 250)
        x = self.relu(self.bn_temp(self.temp_conv(x))) # (B, 64, 125)
        x = self.relu(self.bn_spat(self.spat_conv(x))) # (B, 128, 125)
        x = self.pool(x)                               # (B, 128, 64)
        
        # Reshape for transformer: (B, SeqLen=64, EmbedDim=128)
        x = x.transpose(1, 2)
        x = self.transformer(x)
        
        # Flatten and project to latent vector
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    Modulates visual feature maps using scale (gamma) and shift (beta) parameters predicted from the EEG vector.
    """
    def __init__(self, feature_channels, cond_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim // 2),
            nn.ReLU(),
            nn.Linear(cond_dim // 2, 2 * feature_channels)
        )
        
    def forward(self, x, cond):
        # x: (B, C, H, W) visual features
        # cond: (B, cond_dim) EEG condition vector
        params = self.mlp(cond).unsqueeze(-1).unsqueeze(-1) # (B, 2*C, 1, 1)
        gamma, beta = torch.chunk(params, 2, dim=1)         # Split into scale and shift
        return x * (1 + gamma) + beta                       # Modulate

class BrainGazeDiffusionModel(nn.Module):
    """
    BrainGaze-Diffusion (BGD) Hybrid Model:
    Combines frozen ImageNet feature priors with an EEG Attention Modulation vector to output saliency maps.
    Uses multi-scale visual skip connections (U-Net style) to achieve pixel-precise resolution.
    """
    def __init__(self, embed_dim=512):
        super().__init__()
        # 1. EEG Encoder
        self.eeg_encoder = EEGAttentionEncoder(embed_dim=embed_dim)
        
        # 2. Multi-Scale Image Prior Encoder (ResNet-18)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.early_layers = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        ) # -> B, 64, 56, 56
        self.layer1 = resnet.layer1  # B, 64, 56, 56
        self.layer2 = resnet.layer2  # B, 128, 28, 28
        self.layer3 = resnet.layer3  # B, 256, 14, 14
        
        # Freeze visual layers
        for layer in [self.early_layers, self.layer1, self.layer2, self.layer3]:
            for param in layer.parameters():
                param.requires_grad = False
                
        self.spatial_dropout = nn.Dropout2d(p=0.5)
        
        # 3. Learnable Subject Embedding
        self.subject_embed = nn.Embedding(20, embed_dim)
        nn.init.normal_(self.subject_embed.weight, std=0.01)
        
        # 4. FiLM Modulation Layers
        self.film1 = FiLMLayer(256, cond_dim=embed_dim)
        self.film2 = FiLMLayer(128, cond_dim=embed_dim)
        self.film3 = FiLMLayer(64, cond_dim=embed_dim)
        
        # 5. U-Net Skip Decoder
        # Block 1: Upsample layer3 (14x14) -> 28x28
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        # Skip connection 1: Concat(upconv1, layer2) -> 128 + 128 = 256. Conv merges back to 128.
        self.conv_skip1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn_up1 = nn.BatchNorm2d(128)
        
        # Block 2: Upsample -> 56x56
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        # Skip connection 2: Concat(upconv2, layer1) -> 64 + 64 = 128. Conv merges back to 64.
        self.conv_skip2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn_up2 = nn.BatchNorm2d(64)
        
        # Block 3: Upsample -> 112x112
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn_up3 = nn.BatchNorm2d(32)
        
        # Block 4: Upsample -> 224x224
        self.final_up = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.bn_final = nn.BatchNorm2d(16)
        
        self.conv_out = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, eeg, image, subject_ids=None, zero_image=False):
        # 1. Extract EEG modulation vector
        eeg_feat = self.eeg_encoder(eeg) # (B, embed_dim)
        if subject_ids is not None:
            subject_feat = self.subject_embed(subject_ids) # (B, embed_dim)
            eeg_feat = eeg_feat + subject_feat
            
        # 2. Extract multi-scale visual features
        with torch.no_grad():
            feat0 = self.early_layers(image)
            feat1 = self.layer1(feat0)  # (B, 64, 56, 56)  [Skip 2]
            feat2 = self.layer2(feat1)  # (B, 128, 28, 28) [Skip 1]
            feat3 = self.layer3(feat2)  # (B, 256, 14, 14)
            
        # Apply strict zero image or Modality Dropout
        if zero_image or (self.training and torch.rand(1).item() < 0.5):
            feat1 = torch.zeros_like(feat1)
            feat2 = torch.zeros_like(feat2)
            feat3 = torch.zeros_like(feat3)
        elif self.training:
            feat3 = self.spatial_dropout(feat3)
            
        # 3. Decode with Skip Connections
        # Block 1: 14x14 -> 28x28
        x = self.film1(feat3, eeg_feat)
        x = self.upconv1(x)
        x = torch.cat([x, feat2], dim=1) # Skip connection
        x = self.relu(self.bn_up1(self.conv_skip1(x)))
        
        # Block 2: 28x28 -> 56x56
        x = self.film2(x, eeg_feat)
        x = self.upconv2(x)
        x = torch.cat([x, feat1], dim=1) # Skip connection
        x = self.relu(self.bn_up2(self.conv_skip2(x)))
        
        # Block 3: 56x56 -> 112x112
        x = self.film3(x, eeg_feat)
        x = self.relu(self.bn_up3(self.upconv3(x)))
        
        # Final block: 112x112 -> 224x224
        x = self.relu(self.bn_final(self.final_up(x)))
        out = self.sigmoid(self.conv_out(x))
        return out

if __name__ == "__main__":
    # Test batch
    eeg = torch.randn(2, 32, 250)
    image = torch.randn(2, 3, 224, 224)
    subject_ids = torch.randint(0, 20, (2,))
    model = BrainGazeDiffusionModel()
    out = model(eeg, image, subject_ids)
    print("Model Output Shape with Subject IDs:", out.shape)
