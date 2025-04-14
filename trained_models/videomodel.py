import torch
import cv2
import numpy as np
import face_recognition
from torchvision import transforms, models
from torch import nn
import os
import matplotlib.pyplot as plt
import tempfile
import uuid
import os


class Model(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=False)  
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


def load_video_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  
    return model, device

def preprocess_video(video_path, frame_count=100):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(total_frames // frame_count, 1)

    for i in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(frame)
        
        if face_locations:
            top, right, bottom, left = face_locations[0]
            face = frame[top:bottom, left:right]
        else:
            face = frame  
        
        face = cv2.resize(face, (224, 224))

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        frames.append(transform(face))

    cap.release()
    if not frames:
        return None  
    
    frames = torch.stack(frames)  
    return frames.unsqueeze(0)


def predict_video(model, device, video_path):
    frames = preprocess_video(video_path)
    if frames is None:
        print("No frames extracted. Check video format.")
        return

    frames = frames.to(device)  
    print(f"Input shape to model: {frames.shape}")  

    with torch.no_grad():
        _, output = model(frames)  
        probs = torch.softmax(output, dim=1)  
        confidence, predicted_class = torch.max(probs, dim=1)  

    label = "FAKE" if predicted_class.item() == 1 else "REAL"
    print(f"Prediction: {label} (Confidence: {confidence.item():.4f})")
    return label




 
    
def generate_video_proof_plot(video_path, model, device, overall_label, frame_count=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(total_frames // frame_count, 1)

    for i in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(frame)
        if face_locations:
            top, right, bottom, left = face_locations[0]
            face = frame[top:bottom, left:right]
        else:
            face = frame

        face = cv2.resize(face, (224, 224))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        frames.append(transform(face))

    cap.release()
    if not frames:
        print("No frames extracted.")
        return None

    
    frames_tensor = torch.stack(frames).unsqueeze(0).to(device)
    batch_size, seq_length, c, h, w = frames_tensor.shape

    
    with torch.no_grad():
        x = frames_tensor.view(batch_size * seq_length, c, h, w)
        fmap = model.model(x)
        pooled = model.avgpool(fmap)
        pooled = pooled.view(batch_size, seq_length, -1)
        lstm_out, _ = model.lstm(pooled)
        logits = model.linear1(lstm_out)
        probs = torch.softmax(logits, dim=-1)

    
    confidences = []
    for i in range(seq_length):
        prob = probs[0, i, :].cpu().numpy()
        confidences.append(float(np.max(prob)))

    frame_indices = np.arange(len(confidences))
    color = "#66bb6a" if overall_label.upper() == "REAL" else "#ee0a26"

   
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)


    axs[0].plot(frame_indices, confidences, color=color, marker='o', linestyle='-')
    axs[0].set_title("Confidence Over Time")
    axs[0].set_ylabel("Confidence")
    axs[0].grid(True)
    axs[0].set_ylim(0, 1)

  
    axs[1].bar(frame_indices, confidences, color=color)
    axs[1].set_title("Confidence Per Frame")
    axs[1].set_xlabel("Frame Number")
    axs[1].set_ylabel("Confidence")
    axs[1].set_ylim(0, 1)

    fig.suptitle(
        "Video-Level Detection Breakdown",
        fontsize=16,
        fontweight='bold',
        color="#004830"
    )

    plt.tight_layout()

    # Save to temp path
    temp_dir = tempfile.gettempdir()
    filename = f"video_confidence_dual_{uuid.uuid4().hex}.png"
    save_path = os.path.join(temp_dir, filename)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    return save_path