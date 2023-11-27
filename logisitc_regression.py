import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()
        self.weight = torch.nn.Linear(28 * 28, 10, bias=True)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  
        x = self.weight(x)
        return x

lr = 0.01
weight_decay = 0.0001
epochs = 10
batch_size = 64

# Cuda++
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Tst SGD VS Adam
model_sgd = Regression().to(device)
model_adam = Regression().to(device)

optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=lr, weight_decay=weight_decay)
optimizer_adam = optim.Adam(model_adam.parameters(), lr=lr, weight_decay=weight_decay)

loss_function = torch.nn.MSELoss()


train_accuracy_sgd, test_accuracy_sgd = [], []
train_accuracy_adam, test_accuracy_adam = [], []

for epoch in range(1, epochs + 1):
    model_sgd.train()
    model_adam.train()
    
    train_loss_sgd, train_loss_adam = 0.0, 0.0
    correct_sgd, correct_adam = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Train SGD
        optimizer_sgd.zero_grad()
        logits_sgd = model_sgd(images)
        target = F.one_hot(labels, 10).float()
        loss_sgd = loss_function(logits_sgd, target)
        loss_sgd.backward()
        optimizer_sgd.step()
        train_loss_sgd += loss_sgd.item()
        correct_sgd += (torch.argmax(logits_sgd, dim=1) == labels).sum().item()

        # Train Adam
        optimizer_adam.zero_grad()
        logits_adam = model_adam(images)
        loss_adam = loss_function(logits_adam, target)
        loss_adam.backward()
        optimizer_adam.step()
        train_loss_adam += loss_adam.item()
        correct_adam += (torch.argmax(logits_adam, dim=1) == labels).sum().item()

    train_accuracy_sgd.append(100. * correct_sgd / len(train_loader.dataset))
    train_accuracy_adam.append(100. * correct_adam / len(train_loader.dataset))

    model_sgd.eval()
    model_adam.eval()
    
    correct_sgd, correct_adam = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Upload: Test SGD
            logits_sgd = model_sgd(images)
            correct_sgd += (torch.argmax(logits_sgd, dim=1) == labels).sum().item()

            # Upload: Test Adam
            logits_adam = model_adam(images)
            correct_adam += (torch.argmax(logits_adam, dim=1) == labels).sum().item()

    test_accuracy_sgd.append(100. * correct_sgd / len(test_loader.dataset))
    test_accuracy_adam.append(100. * correct_adam / len(test_loader.dataset))

    print(f"Epoch {epoch}/{epochs} | SGD Train Loss: {train_loss_sgd/len(train_loader):.4f} | Adam Train Loss: {train_loss_adam/len(train_loader):.4f} | SGD Train Accuracy: {train_accuracy_sgd[-1]:.2f}% | Adam Train Accuracy: {train_accuracy_adam[-1]:.2f}% | SGD Test Accuracy: {test_accuracy_sgd[-1]:.2f}% | Adam Test Accuracy: {test_accuracy_adam[-1]:.2f}%")

# 繪製結果
plt.plot(range(1, epochs + 1), train_accuracy_sgd, label='SGD Train')
plt.plot(range(1, epochs + 1), test_accuracy_sgd, label='SGD Test')
plt.plot(range(1, epochs + 1), train_accuracy_adam, label='Adam Train')
plt.plot(range(1, epochs + 1), test_accuracy_adam, label='Adam Test')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy vs Epochs (SGD vs Adam)')
plt.show()
