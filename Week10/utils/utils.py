import torch
import torch.nn as nn

def masked_loss(label, pred):
    mask = torch.argmax(label, axis=-1) != 0
    loss_object = nn.CrossEntropyLoss(reduction='none')
    loss = loss_object(label, pred)
    mask = mask.type_as(loss)
    loss *= mask

    loss = loss.sum() / mask.sum()
    return loss

def masked_accuracy(label, pred):
    pred = pred.argmax(dim=2)
    label = label.argmax(dim=2)
    label = label.type_as(pred)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = match.type(torch.float32)
    mask = mask.type(torch.float32)
    return match.sum() / mask.sum()