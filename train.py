import os
import logging

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

import config as cfg
import utils

utils.set_logger(os.path.join('log', 'train.log'))


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device):
    model.to(device)

    best_acc = 0
    best_model = None

    for epoch in range(cfg.epochs):
        model.train()

        train_loss = 0
        train_accuracy = 0

        print(f'Epoch {epoch}/{cfg.epochs}')
        print('-' * 70)

        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            prediction = model(data)

            loss = criterion(prediction, label)
            loss.backward()

            optimizer.step()

            acc = (prediction.argmax(dim=1) == label).float().mean()
            train_accuracy += acc / len(train_loader)
            train_loss += loss / len(train_loader)

        with torch.no_grad():
            validation_loss = 0
            validation_accuracy = 0

            for data, label in val_loader:
                data = data.to(device)
                label = label.to(device)

                val_prediction = model(data)

                val_loss = criterion(val_prediction, label)

                val_acc = (val_prediction.argmax(dim=1) == label).float().mean()

                validation_accuracy += val_acc / len(val_loader)
                validation_loss += val_loss / len(val_loader)

        print(
            f"Epoch : {epoch + 1} - loss : {train_loss:.4f} - acc: {train_accuracy:.4f} - val_loss : {validation_loss:.4f} - val_acc: {validation_accuracy:.4f}\n"
        )

        logging.info(f'- Train acc: {str(train_accuracy)}, - Train loss: {str(train_loss)}')
        logging.info(f'- Validation acc: {str(validation_accuracy)}, - Validation loss: {str(validation_loss)}')

        if scheduler is not None:
            scheduler.step()

        if best_acc < val_acc:
            best_acc = val_acc
            best_model = model

    return best_model


def train_distillation(student_model, teacher_model, data_loader, criterion_kd, optimizer, device):

    student_model.to(device)
    teacher_model.to(device)

    for epoch in range(cfg.epochs):
        student_model.train()
        teacher_model.eval()

        train_loss = 0
        train_accuracy = 0

        print(f'Epoch {epoch}/{cfg.epochs}')
        print('-' * 70)

        for data, label in tqdm(data_loader):
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            prediction = student_model(data)

            teacher_prediction = teacher_model(data).detach()

            loss = criterion_kd(prediction, label, teacher_prediction, T=10, alpha=0.1)

            loss.backward()

            optimizer.step()

            acc = (prediction.argmax(dim=1) == label).float().mean()
            train_accuracy += acc / len(data_loader)
            train_loss += loss / len(data_loader)

        with torch.no_grad():
            validation_loss = 0
            validation_accuracy = 0

            for data, label in data_loader:
                data = data.to(device)
                label = label.to(device)

                val_prediction = student_model(data)

                val_loss = F.cross_entropy(val_prediction, label)

                val_acc = (val_prediction.argmax(dim=1) == label).float().mean()

                validation_accuracy += val_acc / len(data_loader)
                validation_loss += val_loss / len(data_loader)

        print(
            f"Epoch : {epoch + 1} - loss : {train_loss:.4f} - acc: {train_accuracy:.4f} - val_loss : {validation_loss:.4f} - val_acc: {validation_accuracy:.4f}\n"
        )
        logging.info(f'- Train acc: {str(train_accuracy)}, - Train loss: {str(train_loss)}')
        logging.info(f'- Validation acc: {str(validation_accuracy)}, - Validation loss: {str(validation_loss)}')

def inference(model, test_loader, device):
    model.to(device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for data in test_loader:
            data = data.float().to(device)

            probs = model(data)
            probs = probs.cpu().detach().numpy()

            preds = probs > 0.5
            preds = preds.astype(int)
            predictions += preds.tolist()
        return predictions
