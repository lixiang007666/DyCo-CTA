import os
import random
import torch
import numpy as np
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
from monai.networks.utils import one_hot

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 3.0
    x = x / temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

# def entropy_minimization(outputs, e_margin):
#     """Calculate entropy of the output of a batch of images.
#     """
#     # convert to probabilities
#     entropys = softmax_entropy(outputs)
#     # filter unreliable samples
#     filter_ids_1 = torch.where(entropys < e_margin)
#     selection_rate = len(filter_ids_1[0]) / entropys.numel()
#     print(f"Selection Rate: {selection_rate:.2%}")
#     # ids1 = filter_ids_1
#     # ids2 = torch.where(ids1[0] > -0.1)
#     entropys = entropys[filter_ids_1]
#     loss = entropys.mean(0)
#     return loss

def entropy_minimization(outputs, selection_ratio=0.3):
    entropys = softmax_entropy(outputs)
    
    flat_entropys = entropys.view(-1)
    k = max(1, int(flat_entropys.numel() * selection_ratio))
    
    threshold, _ = torch.topk(flat_entropys, k, largest=False)
    e_margin = threshold[-1]
    # print(f"e_margin: {e_margin:.8f}")
    mask = torch.where(entropys < e_margin)
    
    loss = entropys[mask].mean()
    return loss


def compute_metrics(pred, target, num_classes=2):
    """
    Compute multiple evaluation metrics for 3D segmentation using MONAI library
    """
    # Apply softmax to get probabilities
    pred = torch.argmax(pred, dim=1).unsqueeze(1)
    pred = one_hot(pred, num_classes=num_classes, dim=1).float()

    target = one_hot(target, num_classes=num_classes, dim=1).float()

    # Initialize MONAI metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    iou_metric = MeanIoU(include_background=False, reduction="mean")

    # Compute metrics
    dice_result = dice_metric(pred, target)
    iou_result = iou_metric(pred, target)
    
    # Initialize distance metrics
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    hd95_result = hd95_metric(pred, target)

    # Convert results to scalar values
    dice_score = dice_result.mean().item() if isinstance(dice_result, torch.Tensor) else dice_result
    iou_score = iou_result.mean().item() if isinstance(iou_result, torch.Tensor) else iou_result
    hd95_score = hd95_result.mean().item() if isinstance(hd95_result, torch.Tensor) else hd95_result

    return {
        'dice': dice_score,
        'iou': iou_score,
        'hd95': hd95_score,
    }

def set_seed(seed=32):
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
