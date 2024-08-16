import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, h1_dim, h2_dim, h3_dim, device):
        super(Encoder, self).__init__()
        self.lstm_cell_1 = nn.LSTMCell(input_dim, h1_dim)
        self.lstm_cell_2 = nn.LSTMCell(h1_dim, h2_dim)
        self.lstm_cell_3 = nn.LSTMCell(h2_dim, h3_dim)
        self.input_dim = input_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.h3_dim = h3_dim
        self.device = device

    def forward(self, input_seq):
        h1, c1 = torch.zeros((input_seq.shape[1], self.h1_dim)).to(self.device), torch.zeros((input_seq.shape[1], self.h1_dim)).to(self.device)
        h2, c2 = torch.zeros((input_seq.shape[1], self.h2_dim)).to(self.device), torch.zeros((input_seq.shape[1], self.h2_dim)).to(self.device)
        h3, c3 = torch.zeros((input_seq.shape[1], self.h3_dim)).to(self.device), torch.zeros((input_seq.shape[1], self.h3_dim)).to(self.device)
        for i in range(len(input_seq)):
            h1, c1 = self.lstm_cell_1(input_seq[i], (h1, c1))
            h2, c2 = self.lstm_cell_2(h1, (h2, c2))
            h3, c3 = self.lstm_cell_3(h2, (h3, c3))
        return h1, c1, h2, c2, h3, c3
