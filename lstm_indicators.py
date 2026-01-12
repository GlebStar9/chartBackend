import ccxt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import ta  # <--- библиотека с RSI, MACD, EMA и др.

# === 1. Загружаем данные ===
exchange = ccxt.binance()
bars = exchange.fetch_ohlcv('BTC/USDT', timeframe='15m', limit=3000)
df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# === 2. Добавляем индикаторы ===
df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
df['ema_fast'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
df['ema_slow'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
macd = ta.trend.MACD(df['close'])
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()

# === 3. Цель — направление следующей свечи ===
df['future'] = df['close'].shift(-1)
df['target'] = (df['future'] > df['close']).astype(int)

# === 4. Убираем пустые значения (после индикаторов) ===
df.dropna(inplace=True)

# === 5. Подготавливаем фичи ===
features = df[['open', 'high', 'low', 'close', 'volume',
               'rsi', 'ema_fast', 'ema_slow', 'macd', 'macd_signal']].values
targets = df['target'].values

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

SEQ_LEN = 50
def create_sequences(data, labels, seq_len=SEQ_LEN):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:i+seq_len]
        y = labels[i+seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X, y = create_sequences(features_scaled, targets)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

# === 6. Модель LSTM ===
class DirectionLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_layers=2):
        super(DirectionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

model = DirectionLSTM()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === 7. Обучение ===
EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x).squeeze()
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

# === 8. Предсказания ===
model.eval()
with torch.no_grad():
    preds = model(X_test).squeeze().numpy()
pred_labels = (preds > 0.5).astype(int)

# === 9. Бэктест стратегии ===
initial_balance = 10000
balance = initial_balance
position = 0
entry_price = 0
fee = 0.001  # комиссия 0.1%

prices = df['close'].values[-len(pred_labels):]

for i in range(len(pred_labels) - 1):
    signal = pred_labels[i]
    price = prices[i]

    # Покупка
    if signal == 1 and position == 0:
        position = 1
        entry_price = price
        balance -= balance * fee

    # Продажа
    elif signal == 0 and position == 1:
        profit = (price - entry_price) / entry_price
        balance *= (1 + profit)
        balance -= balance * fee
        position = 0

if position == 1:
    final_profit = (prices[-1] - entry_price) / entry_price
    balance *= (1 + final_profit)

roi = (balance - initial_balance) / initial_balance * 100
print(f"\n💰 Финальный баланс: {balance:.2f} USDT  (ROI: {roi:.2f}%)")

# === 10. Визуализация ===
plt.figure(figsize=(10,5))
plt.plot(prices[-300:], label='BTC/USDT')
plt.title(f"Backtest ROI: {roi:.2f}%")
plt.legend()
plt.show()