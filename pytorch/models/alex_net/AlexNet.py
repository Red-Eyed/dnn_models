import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import Config


class AlexNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(11, 11), stride=4, padding=2),
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

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc = nn.Sequential(nn.Linear(in_features=6 * 6 * 256, out_features=4096),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(in_features=4096, out_features=4096),
                                nn.ReLU(),
                                nn.Linear(in_features=4096, out_features=num_classes),
                                nn.Dropout()
                                )

    def forward(self, input):
        x = self.conv_net(input)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def train(model: nn.Module, device, train_loader: DataLoader, optimizer: torch.optim, epoch):
    model.train()
    for batch_id, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_id % 10 == 0:
            print(f"batch: {batch_id} epoch: {epoch} loss: {loss.item()}")


if __name__ == '__main__':
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    train_loader = DataLoader(datasets.MNIST(str(Config.DATASETS_DIR),
                                             train=True, download=True,
                                             transform=transforms.Compose([
                                                 transforms.Resize(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,), (0.3081,))
                                             ])),
                              batch_size=32,
                              pin_memory=True)

    num_classes = 10
    model = AlexNet(10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    for epoch in range(1, 10 + 1):
        train(model, device, train_loader, optimizer, epoch)

    torch.save(model.state_dict(), str(Config.MODELS_DIR / "AlexNet"))
