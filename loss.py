import torch
import pdb


def custom_loss(y_true, y_pred):
    # 基础的MSE损失
    mse = torch.mean((y_true - y_pred)**2, dim=1)
    
    # 计算预测的价格变动百分比
    price_change_percentage_predicted = (y_pred[:,1:] - y_true[:,:-1]) / y_true[:,:-1]
    actual = y_true[:,1:] - y_true[:,:-1]
    
    # 根据预测的价格变动百分比制定交易决策:
    # 大于0.1% => 1 (买入或持有)
    # 小于-0.1% => -1 (卖出或空仓)
    # 其他 => 0 (不交易)
    trade_decision_predicted = torch.where(price_change_percentage_predicted > 0.001, 1,
                                           torch.where(price_change_percentage_predicted < -0.001, -1, 0))
    print(trade_decision_predicted)
    # 初始化股份：0-空仓，1-持有股票
    shares = torch.zeros(price_change_percentage_predicted.size(0), dtype=torch.float32).to(y_true.device)  # Make sure to use the same device as your tensors
                                                                                                            
    # 初始化交易结果
    trade_outcome = []
    # pdb.set_trace()
    for i in range(y_true.size(1) - 1):
        shares = (trade_decision_predicted[:,i] == 1) & (shares == 0) | (trade_decision_predicted[:,i] != -1) & (shares == 1)
        trade_outcome.append(shares * actual[:,i])
    trade_outcome = torch.stack(trade_outcome, dim=1)
    
    trade_outcome = trade_outcome.sum(dim=1)
    max_outcome = torch.clamp(actual, min=0).sum(dim=1)
    
    # 计算收益损失：如果交易结果为负（即损失），则增加总损失
    gain_loss = torch.where(max_outcome != 0, (max_outcome - trade_outcome) / max_outcome, torch.ones_like(max_outcome).to(y_true.device))
    gain_loss = torch.clamp(gain_loss, min=0)
    # 总损失包括MSE损失和收益损失，将其平均以得到批次的平均损失
    total_loss = torch.mean(mse + gain_loss)
    
    return total_loss


if __name__ == '__main__':
    a = torch.tensor([[1, 2, 3, 4],[3, 2, 1, 0]], dtype=torch.float)
    b = torch.tensor([[2, 3, 4, 5],[4, 5, 6, 7]], dtype=torch.float)
    l = custom_loss(a, b)
    print(l)
    