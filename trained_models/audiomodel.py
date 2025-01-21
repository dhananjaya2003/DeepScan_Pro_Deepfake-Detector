import os
import numpy as np
import librosa
from tensorflow.keras.models import Model, load_model

def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def shift_pitch(audio, n_steps=2):
    return librosa.effects.pitch_shift(audio, sr=22050, n_steps=n_steps)  

def time_stretch(audio, rate=1.25):
    return librosa.effects.time_stretch(audio, rate=rate)

def extract_features(audio_file, augment=False):
    y, sr = librosa.load(audio_file, sr=None)
    if augment:
        augment_choice = np.random.choice(['noise', 'pitch', 'stretch', None])
        if augment_choice == 'noise':
            y = add_noise(y)
        elif augment_choice == 'pitch':
            y = shift_pitch(y)  # Correctly call shift_pitch without sr
        elif augment_choice == 'stretch':
            y = time_stretch(y)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, max(0, 128 - mel_spec_db.shape[1]))), mode='constant')[:, :128]

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

    features = np.concatenate([ 
        mel_spec_db.flatten(),
        mfccs.mean(axis=1),
        chroma.mean(axis=1),
        spec_contrast.mean(axis=1),
        zero_crossing_rate.mean(axis=1)
    ])
    return features


import numpy as np
import librosa
from tensorflow.keras.models import load_model

def audio_deepfake(audio_file:str, model_path=r"D:\Final Year Project\DeepScan_Pro\trained_models\deepfake_audio_model_24.h5"):
    model = load_model(model_path)
    
    audio_features = extract_features(audio_file)  
    
    audio_features = audio_features.reshape((1, *audio_features.shape, 1)) 
    
    prediction = model.predict(audio_features)
    
    if prediction[0][0] > prediction[0][1]:
        return "Real"
    else:
        return "Fake"


