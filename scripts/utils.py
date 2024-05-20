from torchvision.transforms import transforms
import cv2
from typing import Any
import torch
from PIL import Image
import numpy as np

def get_image(image_path, image_size:dict=(100,100)) -> torch.Tensor:
    """Image read prep for prediction

    Args:
        image_path (Path like): Image path
        image_size (dict, optional): Desired size for transforms. Defaults to (100,100).

    Returns:
        torch.Tensor: Transformed PyTorch Tensor Image
    """
    image = np.array(Image.open(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size))
    ])
    return transformer(image)

