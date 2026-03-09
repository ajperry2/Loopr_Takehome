
import albumentations as A
from albumentations.pytorch import ToTensorV2

from loopr.config.training_nn import TrainingNNConfig

def get_train_transforms(H=TrainingNNConfig.height, W=TrainingNNConfig.width):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        # A.Affine(scale=(0.9,1.1), translate_percent=(0.02,0.02), rotate=(-1,1), p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.RandomBrightnessContrast(p=0.4),
        A.Resize(256, 1600, interpolation=1),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ], is_check_shapes=False)

def get_valid_transforms(H=TrainingNNConfig.height, W=TrainingNNConfig.width):
    return A.Compose([
        A.Resize(256, 1600, interpolation=1),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])
