import torch

for name in ["v3_controlled", "v4_speech_preservation", "v5_balanced", "v6_clean"]:
    path = f"models/tiny_enhancer_{name}.pt"
    m = torch.load(path, map_location='cpu', weights_only=False)
    print(f"{name}: epoch={m['epoch']}, val_loss={m['best_val_loss']:.6f}")
