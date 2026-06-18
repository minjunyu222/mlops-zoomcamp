from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from data import X_train, y_train, X_val, y_val

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_train)
rmse = mean_squared_error(y_train, y_pred,squared=False)
print(f"训练集(1月)上的均方根误差: {rmse:.2f}")
y_pred_val = lr.predict(X_val)
rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
print(f"验证集(2月)上的均方根误差: {rmse_val:.2f}")