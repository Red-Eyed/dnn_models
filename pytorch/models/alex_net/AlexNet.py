import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self, data: torch.Tensor, num_classes: int):
        super().__init__()
        self.conv_net = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=4),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=(3, 3), stride=2),

                                      nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=1,
                                                padding=2),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=(3, 3), stride=2),

                                      nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=1,
                                                padding=1),
                                      nn.ReLU(),

                                      nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=1,
                                                padding=1),
                                      nn.ReLU(),

                                      nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=1),
                                      nn.MaxPool2d(kernel_size=(3, 3), stride=2),

                                      nn.Dropout())
        batch_size = data.shape[0]
        out_shape = self.conv_net(data).view(batch_size, -1).shape[-1]

        fc_neurons = 4096
        self.fc = nn.Sequential(nn.Linear(in_features=out_shape, out_features=fc_neurons),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(in_features=fc_neurons, out_features=fc_neurons),
                                nn.ReLU(),
                                nn.Linear(in_features=fc_neurons, out_features=num_classes),
                                nn.Dropout()
                                )

    def forward(self, input):
        x = self.conv_net(input)
        x = x.view((x.shape[0], -1))
        x = self.fc(x)
        return x


if __name__ == '__main__':
    batch_size = 32
    chanels = 3
    w = 1024
    h = 1024
    num_classes = 10
    data = torch.ones([batch_size, chanels, w, h])
    model = AlexNet(data, 10).forward(data)
    pass
