# EL-Alert
EL-Alert: An Explainable Lightweight AST Model for Military Situational Awareness and Surveillance

This code was adapted from the original dataset repository (https://github.com/kaen2891/military_audio_dataset/tree/main)
The Grad-CAM and Integrated Gradient XAI approaches were used to test the trained models.


## How to use
python explain_ast_model.py --audio_path ./data/MAD_dataset/test/009/0.wav --model_path ./save/military_ast_ce/best.pth --xai_method gradcam
python explain_ast_model.py --audio_path ./data/MAD_dataset/test/009/0.wav --model_path ./save/military_ast_ce/best.pth --xai_method ig
