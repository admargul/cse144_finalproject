import torch.nn as nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights

from src.config import num_classes


def create_vit_model(device, pretrained = True):
    #creates ViT-B/16 model w/ 100 class output head

    if pretrained:
        weights = ViT_B_16_Weights.DEFAULT
    else:
        weights = None

    model = models.vit_b_16(weights = weights)

    #freeze all pretrained parameters, then during frozen training only the new classifier head will update
    for param in model.parameters():
        param.requires_grad = False

    #replace the original ImageNet classifier head
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    model = model.to(device)

    return model


def unfreeze_last_two_blocks(model):
    #fine-tuning by unfreezing last two encoder blocks to train

    #freeze everything
    for param in model.parameters():
        param.requires_grad = False

    #keep the classifier head trainable
    for param in model.heads.head.parameters():
        param.requires_grad = True

    #unfreeze the final two encoder layers
    for block in model.encoder.layers[-2:]:
        for param in block.parameters():
            param.requires_grad = True

    #parameter counts, verifies fine-tuning is not training everything
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    total_params = sum(p.numel() for p in model.parameters())

    print("Trainable parameters:", trainable_params)
    print("Total parameters:", total_params)

    return model