import torch
import torch.nn as nn
import torch.nn.functional as F

def distillation(y_pred, labels, teacher_scores, T, alpha):
    return nn.KLDivLoss()(F.log_softmax(y_pred/T), F.softmax(teacher_scores/T)) * (T*T * 2.0 * alpha) + F.cross_entropy(y_pred, labels) * (1. - alpha)
