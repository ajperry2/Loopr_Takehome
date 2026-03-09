import torch

from loopr.config.training_nn import TrainingNNConfig

def tile_image(image: torch.Tensor) -> list[torch.Tensor]:
    cursor = 0
    images = []
    while cursor < image.shape[2]:
        x_min = cursor
        x_max = min(image.shape[2], cursor + TrainingNNConfig.width)
        y_min = 0
        y_max = min(image.shape[1], TrainingNNConfig.height)
        tile = torch.zeros(3, TrainingNNConfig.height, TrainingNNConfig.width)
        tile[:, :y_max-y_min, :x_max-x_min] = image[:, y_min:y_max, x_min:x_max]
        
        images.append(tile)
        cursor += TrainingNNConfig.width
    return torch.stack(images)
    