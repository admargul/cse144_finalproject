from torchvision import transforms

from src.config import img_size


def get_train_transform():
    #training uses random augmentation to prevent overfitting

    return transforms.Compose([
        
        transforms.Resize((256, 256)),

        #random crop simulates different zooms/positions
        transforms.RandomResizedCrop(img_size, scale = (0.75, 1.0)),

        transforms.RandomHorizontalFlip(p = 0.5),

        #random rotation
        transforms.RandomRotation(15),

        #color jitter for brightness/color variation
        transforms.ColorJitter(
            brightness = 0.25,
            contrast = 0.25,
            saturation = 0.25,
            hue = 0.05
        ),

        transforms.ToTensor(),

        # ImageNet normalization, match the pretrained ViT's expected input distribution
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])


def get_val_transform():
    #deterministic tranform used for val

    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])