"""
Utilities to be used along with the deep model
"""
from typing import Union

import torch
from vision.my_resnet import MyResNet18
from vision.simple_net import SimpleNet
from vision.simple_net_final import SimpleNetFinal
from vision.multilabel_resnet import MultilabelResNet18
from torch import nn


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute the accuracy given the prediction logits and the ground-truth labels

    Args:
        logits: The output of the forward pass through the model.
                for K classes logits[k] (where 0 <= k < K) corresponds to the
                log-odds of class `k` being the correct one.
                Shape: (batch_size, num_classes)
        labels: The ground truth label for each instance in the batch
                Shape: (batch_size)
    Returns:
        accuracy: The accuracy of the predicted logits
                   (number of correct predictions / total number of examples)
    """
    batch_accuracy = 0.0

    # # logits: (N, num_classes), labels: (N,)
    # with torch.no_grad():
    #     preds = torch.argmax(logits, dim=1)
    #     correct = (preds == labels).sum().item()
    #     total = labels.shape[0]
    #     if total == 0:
    #         batch_accuracy = 0.0
    #     else:
    #         batch_accuracy = correct / float(total)

    if labels.numel() == 0:
        return 0.0
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        correct = (preds == labels).sum().item()
        return correct / float(labels.shape[0])


    return batch_accuracy


def compute_loss(
    model: Union[SimpleNet, SimpleNetFinal, MyResNet18, MultilabelResNet18],
    model_output: torch.Tensor,
    target_labels: torch.Tensor,
    is_normalize: bool = True,
) -> torch.Tensor:
    """
    Computes the loss between the model output and the target labels

    Args:
    -   model: a model (which inherits from nn.Module)
    -   model_output: the raw scores output by the net
    -   target_labels: the ground truth class labels
    -   is_normalize: bool flag indicating that loss should be divided by the batch size
    Returns:
    -   the loss value
    """
    loss = None


    # # If model provides a loss_criterion, prefer using it
    # if hasattr(model, "loss_criterion") and getattr(model, "loss_criterion") is not None:
    #     criterion = model.loss_criterion
    # else:
    #     # Infer loss type from target shape
    #     if target_labels.dim() == 1 or (target_labels.dim() == 2 and target_labels.size(1) == 1):
    #         # single-label classification
    #         criterion = nn.CrossEntropyLoss(reduction="sum")
    #     else:
    #         # multi-label classification
    #         criterion = nn.BCEWithLogitsLoss(reduction="sum")

    # # compute raw loss (sum reduction to allow optional normalization)
    # # For CrossEntropyLoss the model_output should be raw scores (N, C) and
    # # targets should be (N,) with class indices.
    # loss = criterion(model_output, target_labels)

    # if is_normalize:
    #     # normalize by batch size 
    #     batch_size = target_labels.size(0)
    #     if batch_size > 0:
    #         loss = loss / float(batch_size)

        # Prefer model-provided criterion if available
    criterion = getattr(model, "loss_criterion", None)

    if criterion is None:
        # Infer loss type from target shape:
        # (N,) or (N,1) -> single-label CE; (N, C) -> multi-label BCE
        if target_labels.dim() == 1 or (target_labels.dim() == 2 and target_labels.size(1) == 1):
            # Cross-entropy expects class indices (Long)
            target = target_labels.view(-1).long()
            # Use 'sum' then optionally normalize to mimic 'mean' when requested
            criterion = nn.CrossEntropyLoss(reduction="sum")
            loss = criterion(model_output, target)
            if is_normalize:
                bs = target.shape[0]
                if bs > 0:
                    loss = loss / float(bs)
        else:
            # Multi-label case: targets shape (N, C) with {0,1} floats
            target = target_labels.float()
            criterion = nn.BCEWithLogitsLoss(reduction="sum")
            loss = criterion(model_output, target)
            if is_normalize:
                bs = target.shape[0]
                if bs > 0:
                    loss = loss / float(bs)
    else:
        # Using model's loss_criterion (likely 'mean' reduction per spec)
        # -> Do NOT re-normalize; ignore is_normalize here.
        # Make sure targets have the right dtype/shape for the given criterion.
        if isinstance(criterion, nn.CrossEntropyLoss):
            target = target_labels.view(-1).long()
        elif isinstance(criterion, nn.BCEWithLogitsLoss):
            target = target_labels.float()
        else:
            target = target_labels
        loss = criterion(model_output, target)


    return loss

def compute_multilabel_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute the accuracy given the prediction logits and the ground-truth labels

    Args:
        logits: The output of the forward pass through the model.
                for K labels logits[k] (where 0 <= k < K) corresponds to the
                log-odds of label `k` being present in the image.
                Shape: (batch_size, num_labels)
        labels: The ground truth label for each instance in the batch
                Shape: (batch_size, num_labels)
    Returns:
        accuracy: The accuracy of the predicted logits
                  (number of correct predictions / total number of labels)
    """
    batch_accuracy = 0.0

    if logits.numel() == 0 or labels.numel() == 0:
        return batch_accuracy

    # Convert targets to {0,1} float (robust if labels come as ints or probs)
    targets = (labels > 0.5).to(dtype=torch.float32)

    # # Sigmoid -> probabilities -> threshold at 0.5
    # probs = torch.sigmoid(logits)
    # preds = (probs >= 0.5).to(dtype=torch.float32)
    preds = (logits >= 0.5).to(dtype=torch.float32)

    # Micro accuracy over all entries
    correct = (preds == targets).sum().item()
    total = targets.numel()
    batch_accuracy = correct / float(total)

    return batch_accuracy


def save_trained_model_weights(
    model: Union[SimpleNet, SimpleNetFinal, MyResNet18, MultilabelResNet18], out_dir: str
) -> None:
    """Saves the weights of a trained model along with class name

    Args:
    -   model: The model to be saved
    -   out_dir: The path to the folder to store the save file in
    """
    class_name = model.__class__.__name__
    state_dict = model.state_dict()

    assert class_name in set(
        ["SimpleNet", "SimpleNetFinal", "MyResNet18", "MultilabelResNet18"]
    ), "Please save only supported models"

    save_dict = {"class_name": class_name, "state_dict": state_dict}
    torch.save(save_dict, f"{out_dir}/trained_{class_name}_final.pt")
