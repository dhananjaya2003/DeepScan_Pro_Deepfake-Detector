import flet as ft
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tempfile
import os
import uuid

def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, max(0, 128 - mel_spec_db.shape[1]))), mode='constant')[:, :128]
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]  
    features = np.concatenate([
        mel_spec_db.flatten(),
        mfccs.mean(axis=1),
        chroma.mean(axis=1),
        spec_contrast.mean(axis=1),
        [zero_crossing_rate.mean()]  
    ])
    
    return mfccs, spec_contrast, chroma, zero_crossing_rate, y, sr, mel_spec_db

def make_prediction(audio_path):
    """Runs inference on the uploaded audio file using the trained model."""
    
    features, mfcc, spec_contrast, chroma, zcr, y, sr, mel_spec_db = extract_features(audio_path)

    # Ensure correct shape for model input
    features = features.reshape((1, -1, 1, 1))  # Match model's expected shape

    prediction = model.predict(features)
    
    predicted_label = "Fake" if np.argmax(prediction) == 1 else "Real"
    confidence = prediction[0][0] if predicted_label == "Fake" else 1 - prediction[0][0]
    
    return predicted_label, confidence, mfcc, spec_contrast, chroma, zcr, mel_spec_db

def generate_combined_plot(mel_spec_db, mfcc, chroma, zcr):
    """Generates a 2x2 grid with only the required four audio feature plots and saves it as an image."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 2x2 grid layout

    # Mel-Spectrogram Plot
    librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', ax=axes[0, 0], cmap='magma')
    axes[0, 0].set_title("Mel Spectrogram")

    # MFCC Plot
    librosa.display.specshow(mfcc, x_axis='time', cmap='coolwarm', ax=axes[0, 1])
    axes[0, 1].set_title("MFCC")

    
    librosa.display.specshow(chroma, x_axis='time', cmap='inferno', ax=axes[1, 0])
    axes[1, 0].set_title("Chroma Features")

    axes[1, 1].plot(zcr, color='r')
    axes[1, 1].set_title("Zero Crossing Rate")

    plt.tight_layout()  

    temp_dir = tempfile.gettempdir()
    unique_filename = f"audio_features_{uuid.uuid4().hex}.png"
    img_path = os.path.join(temp_dir, unique_filename)
    plt.savefig(img_path, bbox_inches="tight", dpi=150)
    plt.close(fig)  
    return img_path


def main(page: ft.Page):
    """Flet app for deepfake audio detection."""
    
    page.title = "DeepFake Audio Detection"
    page.window.width = 600
    page.window.height = 700

    # UI Elements
    result_text = ft.Text("Upload an audio file to analyze", size=16)
    confidence_text = ft.Text("", size=14)
    plot_image = ft.Image(src=None)  # Start with no image

    def on_file_upload(e: ft.FilePickerResultEvent):
        """Handles file upload, runs the model, and updates UI with prediction and plots."""
        
        if e.files:
            file_path = e.files[0].path  # Get uploaded file path
            
            # Run prediction
            label, conf, mfcc, spec_contrast, chroma, zcr, mel_spec_db = make_prediction(file_path)
            
           
            result_text.value = f"Prediction: {label}"
            confidence_text.value = f"Confidence: {conf * 100:.2f}%"
            
            
            plot_image.src = None
            page.update()
            
            
            combined_img_path = generate_combined_plot(mel_spec_db, mfcc, chroma, zcr)
            plot_image.src = combined_img_path  
            
            page.update()  
    
    file_picker = ft.FilePicker(on_result=on_file_upload)
    page.overlay.append(file_picker)

    upload_button = ft.ElevatedButton("Upload Audio", on_click=lambda _: file_picker.pick_files(allow_multiple=False))

    page.add(
        ft.Column([
            upload_button,
            result_text,
            confidence_text,
            plot_image
        ], alignment=ft.MainAxisAlignment.CENTER)
    )

ft.app(target=main)