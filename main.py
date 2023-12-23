import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load your data
data = pd.read_csv('2020A.csv')  

# Convert the date column to datetime
data['交易日期'] = pd.to_datetime(data['交易日期'].astype(str), format='%Y%m%d')
data.set_index('交易日期', inplace=True)

# select features and target
features = ['开盘价', '最低价', '最高价', '成交量']
target = '收盘价'

# Set length of history
history_days = 60  # Use last 60 days to predict

stocks = data['WIND代码'].unique()

# Initialize ndarrays for inputs and outputs
X_arr = np.empty((0, history_days, len(features)))
y_arr = np.empty((0,))

for stock in stocks:
    stock_data = data[data['WIND代码'] == stock]
    
    # If not enough data for this stock, skip
    if len(stock_data) < history_days + 1:
        continue
    
    # Allocate array for this stock's data
    X_stock = np.empty((len(stock_data) - history_days, history_days, len(features)))
    y_stock = np.empty((len(stock_data) - history_days,))
    
    # Rolling window for data of this stock
    for i in range(len(stock_data) - history_days):
        X_stock[i] = stock_data[features].values[i:i+history_days]
        y_stock[i] = stock_data[target].values[i+history_days]

    # Append this stock's data
    X_arr = np.append(X_arr, X_stock, axis=0)
    y_arr = np.append(y_arr, y_stock, axis=0)

# Now each sample in X_arr is a window of history_days days, and y_arr is the value to predict for the next day
# X_arr.shape will be (num_samples, history_days, num_features), and y_arr.shape will be (num_samples,)

# Fit model (use first 80% data for training, last 20% for testing)
train_size = int(len(X_arr) * 0.8)
model = LinearRegression()
model.fit(X_arr[:train_size].reshape((train_size, -1)), y_arr[:train_size])

# Predict and calculate RMSE
preds = model.predict(X_arr[train_size:].reshape((len(X_arr) - train_size, -1)))
rmse = np.sqrt(mean_squared_error(y_arr[train_size:], preds))
print(f"RMSE for prediction: {rmse}")