from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_eval_transform(
    image_size: int = 224,
    resize_size: int = 256,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
) -> T.Compose:
    """Deterministic transform used for validation, test and serving."""
    return T.Compose(
        [
            T.Resize(resize_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )


def build_train_transform(
    image_size: int = 224,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
) -> T.Compose:
    """Augmentation: training only, never in the eval or serving path."""
    return T.Compose(
        [
            T.RandomResizedCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )