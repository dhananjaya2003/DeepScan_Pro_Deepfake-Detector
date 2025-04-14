import os
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch.nn.functional as F

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model_wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model_wav2vec = model_wav2vec.to(device)
model_wav2vec.eval()



def extract_embedding(audio_path):
    speech, sr = torchaudio.load(audio_path)
    if speech.shape[0] > 1:
        speech = torch.mean(speech, dim=0, keepdim=True)  
    speech = torchaudio.functional.resample(speech, sr, 16000).squeeze(0)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model_wav2vec(inputs.input_values.to(device)).last_hidden_state
    return outputs.squeeze(0).cpu().numpy()


class AudioDeepfakeDetector(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)  
        attn_weights = F.softmax(self.attn(lstm_out), dim=1) 
        attn_output = torch.sum(attn_weights * lstm_out, dim=1)  
        return self.classifier(attn_output)
    
model = AudioDeepfakeDetector().to(device)
model.load_state_dict(torch.load(r"C:\Users\hp\Downloads\best_model.pth", map_location=torch.device('cpu')))
model.eval()

def audio_deepfake(file_path):
    emb = extract_embedding(file_path)  
    emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)  
    with torch.no_grad():
        output = model(emb_tensor)
        pred = torch.argmax(output, dim=1).item()
        return "Real" if pred == 1 else "Fake"
