import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class StockDataset(Dataset):
    def __init__(self, data, stocks, features, target, history_days, pre_days):
        self.data = data
        self.stocks = stocks
        self.features = features
        self.target = target
        self.history_days = history_days
        self.pre_days = pre_days

    def __len__(self):
        return len(self.stocks)

    def __getitem__(self, idx):
        stock = self.stocks[idx]
        stock_data = self.data[self.data['WIND代码'] == stock]
        
        if len(stock_data) < self.history_days + self.pre_days:
            raise ValueError(f'Not enough data for stock {stock}')
        
        x = stock_data[self.features].values[-(self.history_days+self.pre_days):-self.pre_days]
        y = stock_data[self.target].values[-self.pre_days:]
        
        return x, y


if  __name__ == '__main__':
    # Load data
    data = pd.read_csv('2020A.csv')

    # Convert the date column to datetime
    data['交易日期'] = pd.to_datetime(data['交易日期'].astype(str), format='%Y%m%d')
    data.set_index('交易日期', inplace=True)

    # Select features and target
    features = ['开盘价', '最低价', '最高价', '成交量']
    target = '收盘价'

    # Set length of history and forecast
    history_days = 60  # Use last 60 days to predict
    pre_days = 10  # Predict the next ten days

    stocks = data['WIND代码'].unique()

    # Create dataset and dataloader
    stocks_dataset = StockDataset(data, stocks, features, target, history_days, pre_days)
    stocks_loader = DataLoader(stocks_dataset, batch_size=32, shuffle=True)
    for x in stocks_loader:
        print(x)