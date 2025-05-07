import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 1. 数据加载与预处理
df = pd.read_csv('assignment3data.csv', parse_dates=['date'], index_col='date')

# 2. 单独处理目标变量
target_scaler = StandardScaler()
df['pollution'] = target_scaler.fit_transform(df[['pollution']])

# 3. 处理分类变量（风向）
encoder = OneHotEncoder(sparse_output=False)
wind_encoded = encoder.fit_transform(df[['wnd_dir']])
wind_df = pd.DataFrame(wind_encoded, 
                      columns=encoder.get_feature_names_out(['wnd_dir']), 
                      index=df.index)
df = pd.concat([df.drop(['wnd_dir', 'pollution'], axis=1), wind_df, df['pollution']], axis=1)

# 4. 处理缺失值
df = df.interpolate(method='time').fillna(method='bfill')

# 5. 标准化其他特征
feature_columns = df.columns[:-1]  # 最后列是已标准化的PM2.5
feature_scaler = StandardScaler()
df[feature_columns] = feature_scaler.fit_transform(df[feature_columns])

# 6. 创建监督学习数据集
def create_dataset(data, n_steps=3):
    X, y = [], []
    for i in range(len(data)-n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps, -1])  # 最后列是PM2.5
    return np.array(X), np.array(y)

n_steps = 6
X, y = create_dataset(df.values, n_steps)

# 7. 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, shuffle=False)

# 8. 构建LSTM模型
model = Sequential()
model.add(LSTM(64, activation='relu', 
              return_sequences=True, 
              input_shape=(n_steps, X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 9. 训练模型
early_stop = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(X_train, y_train, 
                   epochs=50, 
                   batch_size=32,
                   validation_data=(X_val, y_val),
                   callbacks=[early_stop],
                   verbose=1)

#模型评估
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.show()

# 10. 预测与反标准化
test_pred = model.predict(X_test)
y_pred = target_scaler.inverse_transform(test_pred)
y_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))

# 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(y_actual, label='Actual PM2.5')
plt.plot(y_pred, label='Predicted PM2.5')
plt.legend()
plt.xlabel('Time Steps')
plt.ylabel('PM2.5 Concentration')
plt.title('Actual vs Predicted PM2.5')
plt.show()

# 计算指标
mse = np.mean((y_actual - y_pred)**2)
mae = np.mean(np.abs(y_actual - y_pred))
print(f'Test MSE: {mse:.2f}, MAE: {mae:.2f}')

# 绘制双指标曲线（MSE和MAE）
plt.figure(figsize=(15, 6))


# 训练MAE曲线
plt.subplot()
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE Progression')
plt.ylabel('MAE')
plt.xlabel('Epochs')
plt.legend()

plt.tight_layout()
plt.show()