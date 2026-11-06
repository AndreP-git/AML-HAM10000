from torchvision import transforms

# ImageNet statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalization_for_model(model_name: str, dataset_mean, dataset_std):
    """Pick the right normalization stats for a given model.
    """
    if model_name.lower() == "simple_cnn":
        return dataset_mean, dataset_std
    return IMAGENET_MEAN, IMAGENET_STD


def build_transforms(image_size: int, mean=None, std=None):
    """
    Augmentation choices are chosen specifically for skin lesion classification:
    - Horizontal AND vertical flips + rotation up to 20°: a skin lesion has no
        canonical "up", unlike, say, a chest X-ray, so all orientations are
        equally valid and safe to train on.
    - Mild color jitter: accounts for differences in lighting / camera /
        skin tone, but kept small so we don't destroy diagnostic colour cues.
    """
    mean = mean or IMAGENET_MEAN
    std = std or IMAGENET_STD

    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tf, eval_tf


def denormalize(tensor, mean=None, std=None):
    mean = mean or IMAGENET_MEAN
    std = std or IMAGENET_STD
    out = tensor.clone()
    for channel, m, s in zip(out, mean, std):
        channel.mul_(s).add_(m)
    return out.clamp_(0, 1)
