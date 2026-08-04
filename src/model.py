# ==========================================
# SRC/MODEL.PY - VIDEO SWIN TRANSFORMER MODEL
# ==========================================
import torch
import torch.nn as nn
import torchvision.models.video as video_models


def build_model(config: dict) -> nn.Module:
    """
    Builds and initializes the Video Swin Transformer model.
    Dynamically adjusts the output head to match the total number of classes in WLASL.
    """
    model_name = config.get("model_name", "swin3d_t").lower()
    num_classes = config.get("num_classes", 2000)

    print(f"🏗️ Building model architecture: {model_name} | Target Classes: {num_classes}")

    # 1. Load Pre-trained Video Swin Transformer
    if model_name == "swin3d_t":
        weights = video_models.Swin3D_T_Weights.DEFAULT
        model = video_models.swin3d_t(weights=weights)
    elif model_name == "swin3d_s":
        weights = video_models.Swin3D_S_Weights.DEFAULT
        model = video_models.swin3d_s(weights=weights)
    elif model_name == "swin3d_b":
        weights = video_models.Swin3D_B_Weights.DEFAULT
        model = video_models.swin3d_b(weights=weights)
    else:
        raise ValueError(f"❌ Unsupported model architecture: {model_name}")

    # 2. Replace the classification head for our specific WLASL classes count
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)

    # 3. Initialize the new Head weights using Xavier/Glorot Normal initialization
    nn.init.xavier_normal_(model.head.weight)
    if model.head.bias is not None:
        nn.init.constant_(model.head.bias, 0)

    print(f"✅ Model initialized with pre-trained weights ({model_name.upper()}) & updated head!")
    return model