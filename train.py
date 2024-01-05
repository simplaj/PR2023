from torch.utils.data import DataLoader
from model import TransformerModel
from data import StockDataset
import torch
import wandb
from loss import custom_loss      # 导入自定义的损失函数
from tqdm import tqdm
import argparse

from utils import save_and_plot_predictions

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input_size', type=int, default=5)
parser.add_argument('--d_model', type=int, default=256)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--lr', type=int, default=1e-3)
parser.add_argument('--seq_len', type=int, default=90)
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--re_path', type=str, default=None)

args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args.device = device

wandb.init(
    project="stock",
    config=args
)

if not args.re_path:
    model = TransformerModel(args)
    re_epoch = -1
else:
    model = torch.load(args.re_path)
    re_epoch = int(args.re_path.split('_')[-2])
    
model.to(device)

criterion = torch.nn.MSELoss()  # 用于回归的你定义的损失函数
optimizer = torch.optim.Adam([{'params': model.parameters(), 
                               'initial_lr': args.lr/25,
                               'max_lr': args.lr,
                               'max_momentum': 0.95,
                               'base_momentum':0.85,
                               'min_lr':0}], lr=args.lr,)  # Adam 优化器

stocks_dataset = StockDataset()
stocks_loader = DataLoader(stocks_dataset, batch_size=args.bs, shuffle=True)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                args.lr,
                args.epochs,
                steps_per_epoch=len(stocks_loader),
                last_epoch=re_epoch,
            )

for epoch in tqdm(range(re_epoch + 1, args.epochs)):
    train_loss = []
    all_predictions = []
    all_labels = []
    for batch_idx, (src, tgt) in enumerate(stocks_loader, 0):
        src, tgt = src.to(device).float(), tgt.to(device).float()
        tgt = torch.cat([src[:, -1:, :], tgt], dim=1)
        tgt_mask = torch.tril(torch.ones(tgt.size(1), tgt.size(1)), diagonal=0) == 0
        #
        tgt_mask = tgt_mask.to(device)
        optimizer.zero_grad()
        y_pred = model(src, tgt, tgt_mask)
        loss = criterion(y_pred[:, :-1, -1], tgt[:, 1:, -1])
        all_predictions.extend(y_pred[:, :-1, -1].detach().cpu().numpy())
        all_labels.extend(tgt[:, 1:, -1].detach().cpu().numpy())
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch} Batch {batch_idx} loss: {sum(train_loss) / len(train_loss):.2f}')

    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    wandb.log({
        'lr':lr,
        'train loss':sum(train_loss) / len(train_loss),
    })
    # Save and plot the predictions after each epoch
    save_and_plot_predictions(all_predictions, all_labels, epoch)
    if epoch % 10 == 0:
        torch.save(model, f'./train_results/model_{epoch}_{loss.item()}.pth')

    if (epoch+1) % 1 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, args.epochs, loss.item()))