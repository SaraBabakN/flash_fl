from torch import nn
import torch.nn.functional as F


class MNISTNet(nn.Module):

    def __init__(self, *args, **kwargs):
        super(MNISTNet, self).__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(1, 10, 5)  # "Conv 1-10"
        self.conv2 = nn.Conv2d(10, 20, 5)  # "Conv 10-20"

        self.fc1 = nn.Linear(20 * 16 * 16, 50)
        self.fc2 = nn.Linear(50, 10)

        # self.init_param_sizes()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, stride=1))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, stride=1))
        x = x.view(-1, 20 * 16 * 16)  # flatten
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


class leaf_cnn(nn.Module):
    def __init__(self, outpusize):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(3136, 2048)
        self.fc2 = nn.Linear(2048, outpusize)

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.fc2(x)
        return x

    def get_embedding(self, x):
        x = x.view((x.shape[0], 28, 28))
        x = x.unsqueeze(1)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        return x
