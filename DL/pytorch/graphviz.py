import torch
import torch.nn as nn
import torch.nn.functional as F


INPUT_SIZE = 28*28

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x)

        return x


from torchviz import make_dot

INPUT_SIZE = 28*28

model = NeuralNet()
data = torch.randn(1, INPUT_SIZE)

y = model(data)

image = make_dot(y, params=dict(model.named_parameters()))
image.format = "png"
image.render("NeuralNet")