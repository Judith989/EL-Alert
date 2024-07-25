import torch
import torchaudio
import numpy as np
import cv2
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, LayerGradCam
from captum.attr import visualization as viz
import timm
from torch import nn
from torch.nn import functional as F
from timm.models.layers import to_2tuple
import argparse

# Custom PatchEmbed to handle 1-channel input
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

# Preprocess the audio input
def preprocess_audio(audio_path, sample_rate=16000, img_size=(224, 224)):
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=128
    )(waveform).unsqueeze(0)
    spectrogram = F.interpolate(spectrogram, size=img_size)
    return spectrogram

# Load the pretrained model
def load_pretrained_model(model_path, device):
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=7)
    model.patch_embed = PatchEmbed(img_size=224, patch_size=16, in_chans=1, embed_dim=768)
    checkpoint = torch.load(model_path, map_location=device)

    # Handle the 'module.' prefix if the model was saved with nn.DataParallel
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)  # Load the updated state_dict with strict=False
    model = model.to(device)
    model.eval()
    return model

# Grad-CAM
def grad_cam(model, input_spectrogram, target_layer, target_class):
    grad_cam = LayerGradCam(model, target_layer)
    attr = grad_cam.attribute(input_spectrogram, target=target_class)
    attr = attr.detach().cpu().numpy()[0]
    cam = np.maximum(attr, 0)
    cam = cv2.resize(cam, (input_spectrogram.shape[3], input_spectrogram.shape[2]))
    cam -= np.min(cam)
    cam /= np.max(cam)
    return cam

# Integrated Gradients
def explain_integrated_gradients(model, input_spectrogram, target_class):
    ig = IntegratedGradients(model)
    attr, _ = ig.attribute(input_spectrogram, target=target_class, return_convergence_delta=True)
    attr = attr.sum(dim=1, keepdim=True)  # Sum across the channels
    attr = attr.cpu().detach().numpy().squeeze()  # Ensure correct shape
    return attr

def main():
    parser = argparse.ArgumentParser(description='XAI for AST Model')
    parser.add_argument('--audio_path', type=str, required=True, help='Path to the audio file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model')
    parser.add_argument('--xai_method', type=str, choices=['gradcam', 'ig'], required=True, help='XAI method to use')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess audio
    input_spectrogram = preprocess_audio(args.audio_path).to(device)

    # Load model
    model = load_pretrained_model(args.model_path, device)

    # Predict
    with torch.no_grad():
        output = model(input_spectrogram)
    predicted_class = torch.argmax(output, dim=1).item()
    print(f'Predicted Class: {predicted_class}')

    # Convert spectrogram for visualization
    input_spectrogram_vis = input_spectrogram.cpu().numpy().squeeze()

    # Visualize original spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(input_spectrogram_vis, aspect='auto', origin='lower')
    plt.title("Original Spectrogram")
    plt.colorbar()
    plt.show()

    if args.xai_method == 'gradcam':
        # Apply Grad-CAM
        target_layer = model.blocks[-1].norm1  # Example target layer
        cam = grad_cam(model, input_spectrogram, target_layer, predicted_class)

        # Visualize Grad-CAM result
        plt.figure(figsize=(10, 4))
        plt.imshow(cam, cmap='jet', aspect='auto', origin='lower')
        plt.colorbar()
        plt.title("Grad-CAM")
        plt.show()

    elif args.xai_method == 'ig':
        # Apply Integrated Gradients
        attr = explain_integrated_gradients(model, input_spectrogram, predicted_class)

        # Visualize Integrated Gradients result
        plt.figure(figsize=(10, 4))
        plt.imshow(attr, cmap='viridis', aspect='auto', origin='lower')
        plt.colorbar()
        plt.title("Integrated Gradients")
        plt.show()

if __name__ == '__main__':
    main()

