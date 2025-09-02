#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试LSTM预测脚本的依赖包是否正确安装
"""

import sys
print(f"Python版本: {sys.version}")

# 测试基础包
try:
    import numpy as np
    print(f"✓ NumPy版本: {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy安装失败: {e}")

try:
    import pandas as pd
    print(f"✓ Pandas版本: {pd.__version__}")
except ImportError as e:
    print(f"✗ Pandas安装失败: {e}")

try:
    import matplotlib.pyplot as plt
    print(f"✓ Matplotlib版本: {plt.matplotlib.__version__}")
except ImportError as e:
    print(f"✗ Matplotlib安装失败: {e}")

try:
    import sklearn
    print(f"✓ Scikit-learn版本: {sklearn.__version__}")
except ImportError as e:
    print(f"✗ Scikit-learn安装失败: {e}")

# 测试TensorFlow
try:
    import tensorflow as tf
    print(f"✓ TensorFlow版本: {tf.__version__}")
    print(f"  TensorFlow后端: {tf.config.list_physical_devices()}")
    
    # 测试LSTM层
    from tensorflow.keras.layers import LSTM
    print("✓ LSTM层可用")
    
    # 测试模型构建
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    
    # 简单测试模型
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(30, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    print("✓ LSTM模型构建成功")
    
except ImportError as e:
    print(f"✗ TensorFlow安装失败: {e}")
except Exception as e:
    print(f"✗ TensorFlow测试失败: {e}")

# 测试数据预处理
try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    print("✓ Scikit-learn预处理和评估函数可用")
except ImportError as e:
    print(f"✗ Scikit-learn函数导入失败: {e}")

print("\n=== 依赖包检查完成 ===")
print("如果所有项目都显示✓，说明环境配置成功！")
print("如果有✗的项目，请按照错误信息重新安装相应的包。")
