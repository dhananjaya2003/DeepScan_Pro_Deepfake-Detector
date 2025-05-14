import torch
from transformers import AutoImageProcessor, SwinModel
from PIL import Image as pimg
import sys
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import visualization as viz
from captum.attr import Occlusion
import tempfile
import os
import uuid
import torch.nn.functional as F
from captum.attr import Occlusion



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SwinFeatureClassifier(nn.Module):
    def __init__(self, input_dim=768):
        super(SwinFeatureClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
swin = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224").to(device)
swin.eval()


classifier = SwinFeatureClassifier().to(device)
classifier.load_state_dict(torch.load(r"C:\Users\hp\Downloads\90_swin_best_classifier.pt", map_location=device))
classifier.eval()


def predict_image(image_path):
    image = pimg.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        swin_out = swin(**inputs)
        cls_token = swin_out.last_hidden_state[:, 0, :]  # CLS token

    with torch.no_grad():
        prob = classifier(cls_token)
        label = "Fake" if prob.item() > 0.5 else "Real"
        confidence = round(prob.item() * 100, 2)
   
    return label


def generate_occlusion_map(image_path, swin, processor, classifier, label):
    image = pimg.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(swin.device)
    image_tensor = inputs["pixel_values"][0]  

    # Occlusion parameters
    window_size = 20
    stride = 10
    _, H, W = image_tensor.shape
    occlusion_map = torch.zeros((H, W))

    # Baseline score (without occlusion)
    with torch.no_grad():
        baseline_feat = swin(pixel_values=image_tensor.unsqueeze(0))["last_hidden_state"][:, 0, :]
        baseline_prob = classifier(baseline_feat).item()

    for i in range(0, H - window_size, stride):
        for j in range(0, W - window_size, stride):
            occluded = image_tensor.clone()
            occluded[:, i:i + window_size, j:j + window_size] = 0  # blackout patch

            with torch.no_grad():
                feat = swin(pixel_values=occluded.unsqueeze(0))["last_hidden_state"][:, 0, :]
                prob = classifier(feat).item()

            impact = baseline_prob - prob if label == "Fake" else prob - baseline_prob
            occlusion_map[i:i + window_size, j:j + window_size] += impact

    occlusion_map = occlusion_map.cpu().numpy()
    occlusion_map = (occlusion_map - occlusion_map.min()) / (occlusion_map.max() - occlusion_map.min() + 1e-6)

    # Plotting side by side
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(image)
    axs[0].set_title("Original")
    axs[0].axis("off")

    cmap = "Reds" if label == "Fake" else "Greens"
    axs[1].imshow(occlusion_map, cmap=cmap, interpolation="nearest")
    axs[1].set_title("Occlusion Map")
    axs[1].axis("off")

    # Save to temp file
    temp_dir = tempfile.gettempdir()
    filename = f"occlusion_{uuid.uuid4().hex}.png"
    file_path = os.path.join(temp_dir, filename)
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Occlusion map saved at: {file_path}")
    return file_path

from torchvision import transforms
def generate_occlusion_map2(image_path, swin, classifier, processor, device, patch_size=20, stride=10):
    image = pimg.open(image_path).convert("RGB")
    original_inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        original_output = swin(**original_inputs)
        cls_token = original_output.last_hidden_state[:, 0, :]
        base_score = classifier(cls_token).item()

    w, h = image.size
    heatmap = np.zeros((h, w))

    transform = transforms.ToTensor()
    image_tensor = transform(image)

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            occluded_image = image_tensor.clone()
            occluded_image[:, y:y + patch_size, x:x + patch_size] = 0.0  # black patch

            occluded_pil = transforms.ToPILImage()(occluded_image)
            occluded_inputs = processor(images=occluded_pil, return_tensors="pt").to(device)

            with torch.no_grad():
                occluded_output = swin(**occluded_inputs)
                cls_token_occ = occluded_output.last_hidden_state[:, 0, :]
                prob = classifier(cls_token_occ).item()

            # The higher the drop, the more important the region
            drop = base_score - prob
            heatmap[y:y + patch_size, x:x + patch_size] += drop

    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap + 1e-8)

    # Plot original + heatmap side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(image)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(image)
    axs[1].imshow(heatmap, cmap='jet', alpha=0.5)
    axs[1].set_title("Occlusion Map")
    axs[1].axis("off")

    # Save to temporary file
    temp_dir = tempfile.gettempdir()
    file_name = f"occlusion_map_{uuid.uuid4().hex[:8]}.png"
    temp_path = os.path.join(temp_dir, file_name)
    plt.tight_layout()
    plt.savefig(temp_path)
    plt.close()

    print(f"[INFO] Occlusion map saved to: {temp_path}")
    return temp_path


def preprocess_image(img_path):
    image = pimg.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    return image, inputs

def get_model_features(inputs):
    with torch.no_grad():
        return swin(**inputs).pooler_output

def get_cmap_by_label(label):
    return "Greens" if label == "Real" else "Reds"

import numpy as np
from scipy.ndimage import zoom 

def generate_occlusion_map3(img_path, label):
    image, inputs = preprocess_image(img_path)
    img_tensor = inputs['pixel_values'].to(device)
    baseline = torch.zeros_like(img_tensor).to(device)

    def model_wrapper(input_tensor):
        feats = swin(pixel_values=input_tensor).pooler_output
        return classifier(feats)

    occlusion = Occlusion(model_wrapper)
    attributions_occ = occlusion.attribute(
        img_tensor,
        strides=(3, 23, 23),
        sliding_window_shapes=(3, 56, 56),
        baselines=baseline
    )

    cmap = get_cmap_by_label(label)

    attr = attributions_occ[0].cpu().permute(1, 2, 0).detach().numpy()
    orig_size = image.size  
    zoom_factors = (
        orig_size[1] / attr.shape[0],  
        orig_size[0] / attr.shape[1],  
        1  
    )
    resized_attr = zoom(attr, zoom_factors, order=1)  

    fig, _ = viz.visualize_image_attr_multiple(
        resized_attr,
        np.array(image) / 255.0,
        methods=["original_image", "heat_map"],
        signs=["all", "positive"],
        titles=["Original", "Occlusion Map"],
        cmap=cmap,
        show_colorbar=True,
        outlier_perc=1,
        use_pyplot=False
    )

    fig.suptitle(
        "Detection Justification via Occlusion Maps",
        fontsize=16,
        fontweight='bold',
        color="#004830"
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    # Save to a temp image
    temp_dir = tempfile.gettempdir()
    filename = f"occlusion_map_{uuid.uuid4().hex}.png"
    save_path = os.path.join(temp_dir, filename)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    return save_path
