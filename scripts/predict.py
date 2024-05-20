import joblib
from typing import List, Any
import torch
import torch.nn as nn
from sklearn.svm import OneClassSVM
import scripts.utils as utils
import numpy as np

extractor = torch.load('apps/model/resnet18_extractor.pth')
predictor = joblib.load('apps/model/predictor.pkl')


def extract_feature(model, image) -> torch.Tensor:
    """Extract images features using the given model

    Args:
        model (PyTorch Model): Used model to be used to extract the features
        image (Image): Image

    Returns:
        torch.Tensor: Flattened extracted features from an image
    """
    model.eval()
    with torch.no_grad():
        logits = model(image.unsqueeze(0))
        flattened_logits = logits.view(logits.size(0), -1)
    return flattened_logits

def get_prediction(image) -> np.ndarray:
    """Prediction function

    Args:
        image (Path like): Image path

    Returns:
        np.ndarray: Array of predicted
    """
    feat = extract_feature(extractor, utils.get_image(image))
    return np.array(predictor.predict(feat))