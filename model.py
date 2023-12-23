import torch
import torch.nn as nn

class StockPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(StockPredictor, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):  # x: (batch_size, sequence_length, input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))  # out: (batch_size, sequence_length, hidden_dim)

        out = self.fc1(out[:, -1, :])  # out: (batch_size, 20)
        return out
    

if __name__ == '__main__':
    model = StockPredictor(input_dim=5, hidden_dim=64, num_layers=2, output_dim=20)
    a = torch.randn(32, 90, 5)
    b = model(a)
    print(b.shape)