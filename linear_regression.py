from imports import *
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from regression_model import Regression

# loss function
loss_function = torch.nn.MSELoss()
train_accuracy = []
test_accuracy = []

model = Regression()
model.to(device)
# Define the optimizer

optimizer = optim.SGD(model.parameters(),
                      lr=lr,
                      momentum=momentum,
                      weight_decay=weight_decay)

# iterate over epochs
for epoch in range(1, epochs + 1):
    # train phase
    model.train()
    accuracy = 0
    N = 0

    # iterate over train data
    for batch_idx, (images, labels) in enumerate(train_loader, start=1):
        images, labels = images.to(device), labels.to(device)

        # forward pass
        logits = model(images)
        target = F.one_hot(labels, 10).float()
        loss = loss_function(logits, target)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # check if predicted labels are equal to true labels
        predicted_labels = torch.argmax(logits, dim=1)
        accuracy += torch.sum((predicted_labels == labels).float()).item()
        N += images.shape[0]

    print("Epoch: " + str(epoch) + " -- Avg. Accuracy: " + str(100. * accuracy / N))
    train_accuracy.append(100. * accuracy / N)

    # test phase
    model.eval()
    accuracy = 0
    N = 0

    # iterate over test data
    for batch_idx, (images, labels) in enumerate(test_loader, start=1):
        images, labels = images.to(device), labels.to(device)

        # forward pass
        logits = model(images)

        # check if predicted labels are equal to true labels
        predicted_labels = torch.argmax(logits, dim=1)
        accuracy += torch.sum((predicted_labels == labels).float()).item()
        N += images.shape[0]
    test_accuracy.append(100. * accuracy / N)
    print(test_accuracy[-1])

# plot results
plt.title('Accuracy Versus Epoch')
plt.plot(range(1, epochs + 1), train_accuracy, label='Train')
plt.plot(range(1, epochs + 1), test_accuracy, label='Test')
plt.legend()
