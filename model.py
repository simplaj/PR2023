import torch
import torch.nn as nn


class MLP(nn.Module):
    """Some Information about MLP"""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(0.15)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Attention(nn.Module):
    """Some Information about Attention"""
    def __init__(self, head, in_dim):
        super(Attention, self).__init__()
        self.in_dim = in_dim
        self.head = head
        dim = in_dim // head
        
        self.qkv = nn.Linear(in_dim, in_dim * 3)
        self.attn_drop = nn.Dropout(0.2)
        self.porj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(0.2)
        self.sm = nn.Softmax(dim=-1)
        self.scale = dim ** -0.5

    def forward(self, x):
        b, d, c = x.shape
        qkv = self.qkv(x).reshape(b, d, 3, c)\
            .reshape(b, self.head, d//self.head, 3, c).permute(3, 0, 1, 2, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-1, -2)
        attn = self.sm(attn * self.scale)
        attn = self.attn_drop(attn)
        
        x = attn @ v
        x = x.reshape(b, d, c)
        x = self.porj(x)
        x = self.proj_drop(x)
        
        return x


class AttnLayer(nn.Module):
    """Some Information about AttnLayer"""
    def __init__(self, head, in_dim):
        super(AttnLayer, self).__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = Attention(head, in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = MLP(in_dim, in_dim, in_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class StockAttention(nn.Module):
    """Some Information about StockAttention"""
    def __init__(self, in_dim, hid_dim, max_len, head, out_dim, layers):
        super(StockAttention, self).__init__()
        self.stock_emb = nn.Linear(in_dim, hid_dim)
        self.tem_emb = nn.Parameter(torch.zeros(1, max_len, hid_dim))
        self.blocks = nn.ModuleList(
            [
                AttnLayer(head, hid_dim) for _ in range(layers)
            ]
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)
        # self.pooling2 = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Linear(max_len, out_dim)
        # self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.stock_emb(x)
        x = x + self.tem_emb
        for layer in self.blocks:
            x = layer(x)
        # max_x = self.pooling2(x)
        # max_x = torch.flatten(max_x, 1)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x


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
    

class TransformerModel(nn.Module):
    def __init__(self, args):
        super(TransformerModel, self).__init__()
        self.args = args
        self.trans = nn.Linear(args.input_size, args.d_model)
        self.pos_emb = PositionalEncoding(args.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.d_model,
            nhead=args.nhead,
            dim_feedforward=4 * args.d_model,
            batch_first=True,
            dropout=0.2,
            device=device
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=args.d_model,
            nhead=args.nhead,
            dropout=0.2,
            dim_feedforward=4 * args.d_model,
            batch_first=True,
            device=device
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=args.num_layers)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=args.num_layers)

        self.output_fc = nn.Sequential(
            nn.Linear(args.d_model, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, args.input_size)
        )

    def encode(self, src):
        src = self.trans(src)
        src = self.pos_emb(src)
        memory = self.encoder(src)

        return memory

    def decode(self, tgt, memory, tgt_mask):
        tgt = self.trans(tgt)
        tgt = self.pos_emb(tgt)
        out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
        out = self.output_fc(out)

        return out

    def forward(self, src, tgt, tgt_mask):
        memory = self.encode(src)
        out = self.decode(tgt, memory, tgt_mask)

        return out


if __name__ == '__main__':
    # model = StockPredictor(input_dim=5, hidden_dim=64, num_layers=2, output_dim=20)
    model = StockAttention(5, 32, 90, 9, 20, 1)
    a = torch.randn(32, 90, 5)
    b = model(a)
    print(b.shape)