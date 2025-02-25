# EL-Alert
EL-Alert: An Explainable Lightweight AST Model for Military Situational Awareness and Surveillance

This repository provides the code for EL-Alert, an advanced system designed for real-time audio event detection, classification, and explanation using state-of-the-art deep learning techniques. EL-Alert integrates the Audio Spectrogram Transformer (AST) and Convolutional Neural Network (CNN6) models to enhance military situational awareness and surveillance. The system employs Explainable AI (XAI) methods, including Grad-CAM and Integrated Gradients, to provide visual and interpretable explanations for model predictions.

## How to Use

To run the model with Grad-CAM XAI method:
```bash
python explain_ast_model.py --audio_path ./data/MAD_dataset/test/009/0.wav --model_path ./save/military_ast_ce/best.pth --xai_method gradcam
```

To run the model with Integrated Gradients XAI method:

```bash
python explain_ast_model.py --audio_path ./data/MAD_dataset/test/009/0.wav --model_path ./save/military_ast_ce/best.pth --xai_method ig
```

# Results


Here is a sample of the original spectrogram
![Original Spectrogram](https://github.com/Judith989/EL-Alert/blob/main/original_spect.png)

Here is a sample of the Grad-CAM XAI result
![Grad-CAM](https://github.com/Judith989/EL-Alert/blob/main/Gradcam_1.png)

Here is a sample of the IG XAI result
![Integrated Gradients](https://github.com/Judith989/EL-Alert/blob/main/ig_3.png)



# Acknowledgments
This code was adapted from the original dataset repository (https://github.com/kaen2891/military_audio_dataset/tree/main). The Grad-CAM and Integrated Gradient XAI approaches were used to test the trained models.

