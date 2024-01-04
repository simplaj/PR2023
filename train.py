# train.py
from torch.utils.data import DataLoader
from model import StockPredictor, StockAttention
from data import StockDataset
import torch
import wandb
from loss import custom_loss      # 导入自定义的损失函数
from tqdm import tqdm

from utils import save_and_plot_predictions

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Hyperparameters
input_dim = 5    # number of features
hidden_dim = 256  # hidden layer dimension
max_len = 90
head = 9
out_dim = 20   # output dimension
num_layers = 6   # number of hidden layers
num_epochs = 2000
learnning_rate = 1e-3
wandb.init(
    # set the wandb project where this run will be logged
    project="stock",
    # track hyperparameters and run metadata
    config={
        "input_dim" : 5,   # number of features
        'hidden_dim' : 256,  # hidden layer dimension
        'max_len' : 90,
        'head' : 9,
        'output_dim' : 20,  # output dimension
        'num_layers' : 6,   # number of hidden layers
        'num_epochs' : 100,
    }
)
# Create model
# model = StockPredictor(input_dim, hidden_dim, num_layers, output_dim)
model = StockAttention(input_dim, hidden_dim, max_len, head, out_dim, num_layers)
model.to(device)

# Loss and optimizer
# criterion = custom_loss  # 用于回归的你定义的损失函数
criterion = torch.nn.MSELoss()  # 用于回归的你定义的损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learnning_rate)  # Adam 优化器

# Load data
stocks_dataset = StockDataset()
stocks_loader = DataLoader(stocks_dataset, batch_size=32, shuffle=True)

# Train model

scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                learnning_rate,
                num_epochs,
                steps_per_epoch=len(stocks_loader)
            )
for epoch in range(num_epochs):
    all_predictions = []
    all_labels = []
    all_loss = 0
    for i, (x, y) in enumerate(stocks_loader):
        inputs = x.to(device).float()
        labels = y.to(device).float()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Save predictions and labels for each batch
        all_predictions.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        all_loss += loss.item()
        if i % 10 == 0:
            print(f'Epoch {epoch} Batch {i} loss: {all_loss / (i + 1):.2f}')
            wandb.log({
                'train loss':all_loss / (i + 1),
            })

    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    wandb.log({
        'lr':lr,
    })
    # Save and plot the predictions after each epoch
    save_and_plot_predictions(all_predictions, all_labels, epoch)
    
    torch.save(model, f'./train_results/model_{epoch}_{loss.item()}.pth')

    if (epoch+1) % 1 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))