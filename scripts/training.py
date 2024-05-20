import torch

def get_feature(model, loader) -> torch.Tensor:
    """A training function used to extract all images feature from a given dataloader

    Args:
        model (PyTorch Model): Used model to extract the image feature
        loader (PyTorch DataLoader): DataLoader for the dataset

    Returns:
        torch.Tensor: Flattened concatenated image features
    """
    model.eval()
    feat = None
    with torch.no_grad():
        for image in loader:
            logits = model(image)
            if feat is None:
                feat = torch.cat((feat, logits), 0)
            else:
                feat = logits
        flattened_feat = feat.view(feat.size(0), -1)
    return flattened_feat