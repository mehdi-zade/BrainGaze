import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from src.braingaze_diffusion_model import BrainGazeDiffusionModel
from src.eeg_saliency_pipeline import EEGSaliencyDataset

WEIGHTS = os.path.join(PROJECT_ROOT, "outputs", "models", "braingaze_diffusion_model_v1.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "custom_inference")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def predict_custom(image_path):
    print("--- Running Custom Image BrainGaze Inference ---")
    
    # 1. Verify Image Path
    if not os.path.exists(image_path):
        print(f"ERROR: Custom image not found at: {image_path}")
        print("Please place an image file (e.g. JPG or PNG) and pass its path.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load Model
    model = BrainGazeDiffusionModel().to(device)
    if os.path.exists(WEIGHTS):
        model.load_state_dict(torch.load(WEIGHTS, map_location=device))
        print("Loaded trained model weights successfully.")
    else:
        print(f"ERROR: No trained weights found at {WEIGHTS}. Train the model first.")
        return
    model.eval()
    
    # 3. Load a representative EEG trial from the test set to act as the cognitive carrier
    print("Loading reference EEG trial...")
    EEG_DIR = os.path.join(PROJECT_ROOT, "BGD_Dataset", "EEG")
    STIM_ROOT = os.path.join(PROJECT_ROOT, "BGD_Dataset", "images")
    MAPS_ROOT = os.path.join(PROJECT_ROOT, "BGD_Dataset", "maps")
    dataset = EEGSaliencyDataset(EEG_DIR, STIM_ROOT, MAPS_ROOT, split="test")
    
    # Grab subject 1, trial 0 (index 0)
    sample = dataset[0]
    eeg_tensor = sample['eeg'].unsqueeze(0).to(device) # Shape (1, 32, 250)
    subject_id = torch.tensor([sample['subject_id']]).to(device) # Shape (1,)
    
    # 4. Preprocess Custom Image
    print("Processing custom image prior...")
    orig_img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = orig_img.size
    
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = img_transform(orig_img).unsqueeze(0).to(device) # Shape (1, 3, 224, 224)
    
    # 5. Run Model
    print("Generating attention saliency heatmap...")
    with torch.no_grad():
        out = model(eeg_tensor, image_tensor, subject_id)
        
    # 6. Post-process Heatmap
    pred_map = out.squeeze().cpu().numpy() # (224, 224)
    pred_map = (pred_map - pred_map.min()) / (pred_map.max() - pred_map.min() + 1e-8)
    
    # Resize prediction back to original image size
    pred_img = Image.fromarray((pred_map * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    
    # 7. Create Overlay Visualization
    plt.figure(figsize=(10, 5))
    
    # Left: Stimulus
    plt.subplot(1, 2, 1)
    plt.imshow(orig_img)
    plt.title("Original Stimulus Prior")
    plt.axis("off")
    
    # Right: Overlay Saliency
    plt.subplot(1, 2, 2)
    plt.imshow(orig_img)
    plt.imshow(pred_img, cmap='jet', alpha=0.5) # alpha overlay
    plt.title("BrainGaze Attention Overlay")
    plt.axis("off")
    
    out_name = f"bg_custom_{os.path.splitext(os.path.basename(image_path))[0]}.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"\nSuccess! Saliency prediction overlay saved to:")
    print(f"  {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict_custom_image.py <path_to_image>")
    else:
        predict_custom(sys.argv[1])
