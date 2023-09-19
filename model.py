import torch
from torch import nn
from torch.functional import F


class AutoencoderMlp4Layer(nn.Module):

    def __init__(self, n_input=784, n_bottleneck=8, n_output=784):
        super(AutoencoderMlp4Layer, self).__init__()
        n2 = 392
        self.fc1 = nn.Linear(n_input, n2)  # input = 1x784, output = 1x392
        self.fc2 = nn.Linear(n2, n_bottleneck)  # output = 1xn
        self.fc3 = nn.Linear(n_bottleneck, n2)  # output = 1x392
        self.fc4 = nn.Linear(n2, n_output)  # output = 1x784
        self.type = 'MLP4'
        self.input_shape = (1, 28*28)

    def forward(self, x):
        # encoder
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        # decoder
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)

        return x
