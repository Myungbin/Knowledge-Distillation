import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model.teacher import TeacherNet
from model.student import StudentNet
from train import train, train_distillation
import config as cfg
from utils import distillation


device = 'cuda'
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_ds = datasets.MNIST('./data/', train=True, download=False, transform=data_transform)
val_ds = datasets.MNIST('./data/', train=False, download=False, transform=data_transform)

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=cfg.batch_size*2, shuffle=True)

teacher_model = TeacherNet()
student_model = StudentNet()

criterion = nn.CrossEntropyLoss()
teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=cfg.lr)
student_optimizer = optim.Adam(student_model.parameters(), lr=cfg.lr)


teacher_model = train(teacher_model, train_loader, val_loader, criterion, teacher_optimizer, None, device)

train_distillation(student_model, teacher_model, train_loader, distillation, student_optimizer, device)


