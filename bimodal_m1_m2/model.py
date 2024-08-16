import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter


class PropertiesEncoder(nn.Module):
    def __init__(self, input_dim, h1_dim, h2_dim, h3_dim, device):
        super(PropertiesEncoder, self).__init__()
        self.device = device
        self.lstm_cell_1 = nn.LSTMCell(input_dim, h1_dim)
        self.lstm_cell_2 = nn.LSTMCell(h1_dim, h2_dim)
        self.lstm_cell_3 = nn.LSTMCell(h2_dim, h3_dim)
        self.input_dim = input_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.h3_dim = h3_dim

    def forward(self, input_seq):
        h1, c1 = torch.zeros((input_seq.shape[1], self.h1_dim)).to(
            self.device), torch.zeros((input_seq.shape[1], self.h1_dim)).to(self.device)
        h2, c2 = torch.zeros((input_seq.shape[1], self.h2_dim)).to(
            self.device), torch.zeros((input_seq.shape[1], self.h2_dim)).to(self.device)
        h3, c3 = torch.zeros((input_seq.shape[1], self.h3_dim)).to(
            self.device), torch.zeros((input_seq.shape[1], self.h3_dim)).to(self.device)
        for i in range(len(input_seq)):
            h1, c1 = self.lstm_cell_1(input_seq[i], (h1, c1))
            h2, c2 = self.lstm_cell_2(h1, (h2, c2))
            h3, c3 = self.lstm_cell_3(h2, (h3, c3))
        return h1, c1, h2, c2, h3, c3


class RasterEncoder(nn.Module):
    def __init__(self, input_dim=128, in_channels=1, out_channels=4, latent_dim=256):
        super(RasterEncoder, self).__init__()
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


class InferenceModel(nn.Module):
    def __init__(self, properties_encoder, pil_encoder, fusion_output_dim, device):
        super(InferenceModel, self).__init__()
        self.properties_encoder = properties_encoder
        self.pil_encoder = pil_encoder
        self._freeze()
        self.fusion = LowRankFusion(
            rank=4, output_dim=fusion_output_dim, m1_h=128, m2_h=256, device=device)
        self.fc1 = nn.Linear(fusion_output_dim, fusion_output_dim // 2)
        self.fc2 = nn.Linear(fusion_output_dim//2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, modalities):
        properties, pil = modalities
        properties_encoding = self.properties_encoder(properties)[-1]
        pil_encoding = self.pil_encoder(pil)
        fused = self.fusion([properties_encoding, pil_encoding])
        output = self.fc1(fused)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.sigmoid(output)
        return output

    def _freeze(self):
        for param in self.properties_encoder.parameters():
            param.requires_grad = False
        for param in self.pil_encoder.parameters():
            param.requires_grad = False


class LowRankFusion(nn.Module):
    def __init__(self, rank, output_dim, m1_h, m2_h, device):
        super(LowRankFusion, self).__init__()
        self.rank = rank
        self.output_dim = output_dim
        self.m1_h = m1_h
        self.m2_h = m2_h
        self.m1_factor = Parameter(torch.Tensor(
            self.rank, self.m1_h + 1, self.output_dim))
        self.m2_factor = Parameter(torch.Tensor(
            self.rank, self.m2_h + 1, self.output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))
        self.device = device

        xavier_normal_(self.m1_factor)
        xavier_normal_(self.m2_factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, modalities):
        m1, m2 = modalities
        batch_size = m1.shape[0]
        _m1 = torch.cat((m1, torch.autograd.Variable(torch.ones((batch_size, 1)).type(
            torch.float32).to(self.device), requires_grad=False)), dim=1)
        _m2 = torch.cat((m2, torch.autograd.Variable(torch.ones((batch_size, 1)).type(
            torch.float32).to(self.device), requires_grad=False)), dim=1)
        fusion_m1 = torch.matmul(_m1, self.m1_factor)
        fusion_m2 = torch.matmul(_m2, self.m2_factor)
        fusion_zy = fusion_m1 * fusion_m2
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(
            1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        return output
