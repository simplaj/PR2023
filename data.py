import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

class StockDataset(Dataset):
    def __init__(self):
        data = pd.read_csv('/root/autodl-tmp/2020A.csv')

        data = data[data['停牌'] == 0]
        # Convert the date column to datetime
        data['交易日期'] = pd.to_datetime(data['交易日期'].astype(str), format='%Y%m%d')
        data.set_index('交易日期', inplace=True)

        # Select features and target
        features = ['开盘价', '最低价', '最高价', '成交量', '收盘价']
        target = '收盘价'

        # Set length of history and forecast
        history_days = 90  # Use last 60 days to predict
        pre_days = 20  # Predict the next ten days

        stocks = data['WIND代码'].unique()
        
        stock_counts = data['WIND代码'].value_counts()

        stocks = list(stock_counts[stock_counts >= (history_days + pre_days)].index)
        self.data = data
        import random

        # Randomly shuffle the data
        random.seed(42)
        random.shuffle(stocks)

        # Calculate the index marking 80% of the data
        split_idx = int(0.8 * len(stocks))

        # Split the data into training and validation sets
        train_stocks = stocks[:split_idx]
        valid_stocks = stocks[split_idx:]
        self.stocks = train_stocks
        self.features = features
        self.target = target
        self.history_days = history_days
        self.pre_days = pre_days

    def __len__(self):
        return len(self.stocks)

    def __getitem__(self, idx):
        stock = self.stocks[idx]
        stock_data = self.data[self.data['WIND代码'] == stock]
        
        x = stock_data[self.features].values[-(self.history_days+self.pre_days):-self.pre_days]
        y = stock_data[self.target].values[-self.pre_days:]
        
        return x, y


if  __name__ == '__main__':
    stocks_dataset = StockDataset()
    stocks_loader = DataLoader(stocks_dataset, batch_size=32, shuffle=True)
    # for x in stocks_loader:
    #     print(x)