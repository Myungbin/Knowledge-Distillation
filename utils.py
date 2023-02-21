import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

def distillation(y_pred, labels, teacher_scores, T, alpha):
    return nn.KLDivLoss()(F.log_softmax(y_pred/T), F.softmax(teacher_scores/T)) * (T*T * 2.0 * alpha) + F.cross_entropy(y_pred, labels) * (1. - alpha)


def set_logger(log_path):
    """정보를 기록 하도록 logger를 설정하고 log파일을 저장

    model_dir/train.log로 저장

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) log가 저장될 path
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


