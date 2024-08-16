import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim=128, in_channels=1, out_channels=4, latent_dim=256):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding='same')
        self.conv2 = nn.Conv2d(
            out_channels, out_channels * 2, 3, padding='same')
        self.flatten_dim = (input_dim // 4) ** 2 * out_channels * 2
        self.fc1 = nn.Linear(self.flatten_dim, self.flatten_dim//4)
        self.fc2 = nn.Linear(self.flatten_dim//4, latent_dim)
        self.max = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max(x)
        x = x.view(-1, self.flatten_dim)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim=128, in_channels=1, out_channels=4, latent_dim=256):
        super(Decoder, self).__init__()
        self.reshape_dim = (-1, out_channels * 2,
                            input_dim // 4, input_dim // 4)
        flatten_dim = (input_dim // 4) ** 2 * out_channels * 2
        self.fc2 = nn.Linear(latent_dim, flatten_dim//4)
        self.fc1 = nn.Linear(flatten_dim//4, flatten_dim)
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic')
        self.conv2 = nn.Conv2d(
            out_channels * 2, out_channels, 3, padding='same')
        self.conv1 = nn.Conv2d(out_channels, in_channels, 3, padding='same')
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = x.view(self.reshape_dim)
        x = self.upsample(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x
