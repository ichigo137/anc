import torch

m = torch.load('models/tiny_enhancer_v4_speech_preservation.pt', map_location='cpu', weights_only=False)
print(f"Model: {m['model_name']}")
print(f"Epoch: {m['epoch']}")
print(f"Val loss: {m['best_val_loss']:.6f}")
print(f"Dataset: {m['dataset']}")
print(f"Architecture: {m['architecture']}")
print(f"Improvements: {m['improvements']}")
