from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

from src.config import train_dir, test_dir, batch_size, seed
from src.transforms import get_train_transform, get_val_transform


class NumericImageFolder(datasets.ImageFolder):
    #forced numeric mapping according to project specs

    def find_classes(self, directory):
        directory = Path(directory)

        #all folder names inside train/
        classes = [d.name for d in directory.iterdir() if d.is_dir()]

        #sort w/ int(x)
        classes = sorted(classes, key=lambda x: int(x))

        #convert folder name -> integer label
        class_to_idx = {class_name: int(class_name) for class_name in classes}

        return classes, class_to_idx


class AllTestImagesDataset(Dataset):
    #dataset for test images. ex: tensor, "15.jpg"

    def __init__(self, image_paths, transform = None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image_id = image_path.name

        # Convert every image to RGB to prevent errors
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, image_id


def check_label_mapping(dataset):
    #check that class labels are correct

    for i in range(100):
        folder_name = str(i)

        assert dataset.class_to_idx[folder_name] == i, (f"Wrong mapping: folder {folder_name} maps to " f"{dataset.class_to_idx[folder_name]}, but it should map to {i}")

    print("Label mapping is correct")


def create_train_val_loaders():
    #use stratified split instead of random split because the dataset is very small, each class contributes at least 1 img to val

    train_transform = get_train_transform()
    val_transform = get_val_transform()

    #training gets augmentation.
    train_base_dataset = NumericImageFolder(
        root = train_dir,
        transform = train_transform
    )

    #validation gets deterministic preprocessing.
    val_base_dataset = NumericImageFolder(
        root = train_dir,
        transform = val_transform
    )

    check_label_mapping(train_base_dataset)

    # Group image indices by class label.
    label_to_indices = defaultdict(list)
    #ex: label_to_indices[0] = [indices for class 0 images]

    for idx, (_, label) in enumerate(train_base_dataset.samples):
        label_to_indices[label].append(idx)

    train_indices = []
    val_indices = []

    #np random generator with fixed seed for a reproducible split
    rng = np.random.default_rng(seed)

    for label, indices in label_to_indices.items():
        indices = np.array(indices)

        rng.shuffle(indices)

        n_val = max(1, int(0.2 * len(indices)))

        val_indices.extend(indices[:n_val].tolist())
        train_indices.extend(indices[n_val:].tolist())

    #subset allows reusing the original ImageFolder dataset while selecting only train or validation indices
    train_dataset = Subset(train_base_dataset, train_indices)
    val_dataset = Subset(val_base_dataset, val_indices)

    #when using CUDA pin_memory speeds up
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 2,
        pin_memory = pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 2,
        pin_memory = pin_memory
    )

    print("Train size:", len(train_dataset))
    print("Validation size:", len(val_dataset))

    #return full_train_dataset so train.py can save class_to_idx in checkpoint
    return train_loader, val_loader, train_base_dataset


def create_test_loader():
    #create dataloader for test img, files sorted numerically

    val_transform = get_val_transform()

    test_image_paths = sorted(
        test_dir.glob("*.jpg"),
        key = lambda p: int(p.stem)
    )

    test_dataset = AllTestImagesDataset(
        image_paths = test_image_paths,
        transform = val_transform
    )

    pin_memory = torch.cuda.is_available()

    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 2,
        pin_memory = pin_memory
    )

    print("Number of test images:", len(test_image_paths))

    return test_loader, test_image_paths