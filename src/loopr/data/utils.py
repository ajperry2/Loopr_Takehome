import torch

from loopr.config.training_unet import TrainingUnetConfig

def tile_image(image: torch.Tensor) -> list[torch.Tensor]:
    cursor = 0
    images = []
    while cursor < image.shape[2]:
        x_min = cursor
        x_max = min(image.shape[2], cursor + TrainingUnetConfig.width)
        y_min = 0
        y_max = min(image.shape[1], TrainingUnetConfig.height)
        tile = torch.zeros(3, TrainingUnetConfig.height, TrainingUnetConfig.width)
        tile[:, :y_max-y_min, :x_max-x_min] = image[:, y_min:y_max, x_min:x_max]
        
        images.append(tile)
        cursor += TrainingUnetConfig.width
    return torch.stack(images)
    