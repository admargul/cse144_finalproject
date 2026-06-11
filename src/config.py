from pathlib import Path

#project_root points to the main project folder
project_root = Path(__file__).resolve().parents[1]

#local dataset folder
data_root = project_root / "data" / "ucsc-cse-144-spring-2026-final-project"

train_dir = data_root / "train"
test_dir = data_root / "test"
sample_submission_path = data_root / "sample_submission.csv"

#all generated files (checkpoints, submissions) go into outputs
output_dir = project_root / "outputs"
output_dir.mkdir(parents = True, exist_ok = True)

# Frozen ViT checkpoint paths.
# "Frozen" means the pretrained ViT backbone is frozen and only the classifier head trains.
vit_checkpoint_path = output_dir / "best_vit_frozen.pth"
vit_last_path = output_dir / "last_vit_frozen.pth"

#fine-tuned ViT checkpoint paths
vit_ft_path = output_dir / "best_vit_ft.pth"
vit_ft_last_path = output_dir / "last_vit_ft.pth"

#kaggle submission file
submission_path = output_dir / "submission_vit.csv"

#dataset/model settings.
num_classes = 100
img_size = 224
batch_size = 16
seed = 42

frozen_epochs = 15
frozen_patience = 4

ft_epochs = 20
ft_patience = 8

head_lr = 1e-3
encoder_lr = 1e-5
weight_decay = 1e-4
label_smooth = 0.1