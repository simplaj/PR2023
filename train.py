# train.py
from torch.utils.data import DataLoader
from model import StockPredictor
from data import StockDataset
import torch
from loss import custom_loss      # 导入自定义的损失函数
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Hyperparameters
input_dim = 5    # number of features
hidden_dim = 32  # hidden layer dimension
num_layers = 2   # number of hidden layers
output_dim = 20   # output dimension

# Create model
model = StockPredictor(input_dim, hidden_dim, num_layers, output_dim)
model.to(device)

# Loss and optimizer
criterion = custom_loss  # 用于回归的你定义的损失函数
optimizer = torch.optim.Adam(model.parameters())  # Adam 优化器

# Load data
stocks_dataset = StockDataset()
stocks_loader = DataLoader(stocks_dataset, batch_size=32, shuffle=True)

# Train model
num_epochs = 100
for epoch in range(num_epochs):
    for i, (x, y) in tqdm(enumerate(stocks_loader)):
        inputs = x.to(device).float()
        labels = y.to(device).float()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    torch.save(model, f'model_{epoch}_{loss.item()}.pth')

    if (epoch+1) % 1 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))