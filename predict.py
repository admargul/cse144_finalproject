import pandas as pd
import torch

from src.config import (
    vit_ft_path,
    submission_path,
)

from src.utils import get_device
from src.data import create_test_loader
from src.model import create_vit_model
from src.engine import predict_with_tta


def main():
    #create final submission with best pth

    device = get_device()
    print("Using device:", device)

    test_loader, test_image_paths = create_test_loader()

    #pretrained=False used to ommit ImageNet weights
    model = create_vit_model(device, pretrained = False)

    checkpoint = torch.load(vit_ft_path, map_location = device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print("Loaded best fine-tuned ViT checkpoint.")
    print("Best ViT validation accuracy:", checkpoint.get("best_val_acc", checkpoint.get("val_acc", "not saved")))
    print("Saved at fine-tuning epoch:", checkpoint.get("epoch", "not saved"))

    all_ids, all_predictions = predict_with_tta(model, test_loader, device)

    print("Number of IDs:", len(all_ids))
    print("Number of predictions:", len(all_predictions))
    print("First 10 IDs:", all_ids[:10])
    print("First 10 predictions:", all_predictions[:10])

    #id , label cols for kaggle
    submission_df = pd.DataFrame({"ID": all_ids, "Label": all_predictions})

    submission_df.to_csv(submission_path, index = False)

    print("Saved submission to:", submission_path)
    print(submission_df.head())
    print(submission_df.tail())
    print("Submission shape:", submission_df.shape)
    print("Columns:", submission_df.columns.tolist())

if __name__ == "__main__":
    main()