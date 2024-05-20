from torch.utils.data import Dataset
from torchvision.transforms import transforms
from typing import Any
import cv2
import pandas as pd

class training_dataset(Dataset):
    def __init__(self, df, transforms) -> None:
        super().__init__()
        self.df = df
        self.transforms = transforms
    def __len__(self) -> Any:
        return len(self.df)
    def __getitem__(self, index) -> Any:
        return self.transforms(cv2.imread(self.df.iloc[index,0]))