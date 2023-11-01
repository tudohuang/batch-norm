import torch
import torch.nn as nn


class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()
        # Define the model parameters using Pytorch modules here
        self.weight = torch.nn.Linear(28 * 28, 10, bias=True)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Reshaping the data so that it becomes a vector
        # Fill in the rest here
        x = self.weight(x)
        return x
