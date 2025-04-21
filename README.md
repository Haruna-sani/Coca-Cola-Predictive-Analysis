# 📈 Coca-Cola Stock Price Predictive Analysis using LSTM and CNN+GRU-LSTM

This project focuses on forecasting Coca-Cola stock prices using deep learning models — LSTM and a hybrid CNN+GRU-LSTM architecture. The models are evaluated using **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **R-squared (R²)** metrics to determine prediction accuracy.

## 🔍 Objective

To develop accurate time-series prediction models for Coca-Cola stock prices by experimenting with different architectures, learning rates, and training configurations.

---

## 🧠 Models Used

### 1. **LSTM (Long Short-Term Memory)**

- **Optimal Learning Rate**: `0.01`
- **Performance**:
  - R² Score: `0.999`
  - MAE: `0.412`
  - RMSE: `0.607`

### 2. **CNN + GRU-LSTM Hybrid Model**

- **Optimal Learning Rate**: `0.0001`
- **Performance**:
  - R² Score: `0.999`
  - MAE: `0.449`
  - RMSE: `0.701`

---

## ⚙️ Model Configuration

- **Look-back Window**: `70`
- **Epochs**: `50`
- **Batch Size**: `64`
- **Learning Rates Tested**: `0.0001`, `0.001`, `0.01`, `0.1`

---

## 📦 Required Libraries

Make sure the following Python libraries are installed:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Conv2D, MaxPooling2D,
    Flatten, TimeDistributed, Bidirectional, GRU
)
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')
