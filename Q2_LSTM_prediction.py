# 用LSTM预测'2023-07-01', '2023-07-07'期间的需求和聚合的品类价格

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子以确保结果可重现
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data():
    """加载和预处理数据"""
    
    print("正在加载数据...")
    
    # 加载销量数据（使用file_23_r_with_category_consolidated.csv）
    try:
        sales_df = pd.read_csv('file_23_r_with_category_consolidated.csv')
        print(f"销量数据加载成功，共 {len(sales_df)} 条记录")
    except FileNotFoundError:
        print("未找到销量数据文件，请检查文件名")
        return None, None, None
    
    # 加载批发价格数据（使用file_3.csv）
    try:
        wholesale_df = pd.read_csv('file_3.csv')
        print(f"批发价格数据加载成功，共 {len(wholesale_df)} 条记录")
    except FileNotFoundError:
        print("未找到批发价格数据文件，请检查文件名")
        return None, None, None
    
    # 加载损耗率数据（使用file_14.csv）
    try:
        loss_df = pd.read_csv('file_14.csv')
        print(f"损耗率数据加载成功，共 {len(loss_df)} 条记录")
    except FileNotFoundError:
        print("未找到损耗率数据文件，请检查文件名")
        return None, None, None
    
    return sales_df, wholesale_df, loss_df

def prepare_daily_data(sales_df, wholesale_df):
    """准备每日品类级别的数据"""
    
    print("\n正在准备每日品类数据...")
    
    # 转换日期格式
    sales_df['销售日期'] = pd.to_datetime(sales_df['销售日期'])
    if '日期' in wholesale_df.columns:
        wholesale_df['日期'] = pd.to_datetime(wholesale_df['日期'])
    elif '销售日期' in wholesale_df.columns:
        wholesale_df['销售日期'] = pd.to_datetime(wholesale_df['销售日期'])
    
    # 按照Q2.md中的公式计算加权平均价格
    # P_i(t) = Σ(p_j(t) * q_j(t)) / Σ(q_j(t))
    # C_i(t) = Σ(c_j(t) * q_j(t)) / Σ(q_j(t))
    
    # 计算每日每类的加权平均销售单价和批发价格
    daily_sales = sales_df.groupby(['销售日期', '分类名称']).apply(
        lambda x: pd.Series({
            'quantity': x['销量(千克)'].sum(),  # 总销量
            'price': (x['销售单价(元/千克)'] * x['销量(千克)']).sum() / x['销量(千克)'].sum(),  # 加权平均销售单价
            'wholesale_price': (x['批发价格(元/千克)'] * x['销量(千克)']).sum() / x['销量(千克)'].sum()  # 加权平均批发价格
        })
    ).reset_index()
    
    # 重命名列
    daily_sales.columns = ['date', 'category', 'quantity', 'price', 'wholesale_price']
    
    print(f"每日销量数据：{len(daily_sales)} 条记录")
    print(f"品类：{daily_sales['category'].unique()}")
    print("价格计算方式：按销量加权的加权平均")
    
    # 验证加权平均价格计算是否正确
    print("\n=== 验证加权平均价格计算 ===")
    sample_category = daily_sales['category'].iloc[0]
    sample_date = daily_sales['date'].iloc[0]
    
    # 获取原始数据
    original_data = sales_df[(sales_df['分类名称'] == sample_category) & 
                            (sales_df['销售日期'] == sample_date)]
    
    if len(original_data) > 0:
        # 手动计算加权平均
        manual_price = (original_data['销售单价(元/千克)'] * original_data['销量(千克)']).sum() / original_data['销量(千克)'].sum()
        manual_wholesale = (original_data['批发价格(元/千克)'] * original_data['销量(千克)']).sum() / original_data['销量(千克)'].sum()
        
        # 获取计算后的数据
        calculated_data = daily_sales[(daily_sales['category'] == sample_category) & 
                                    (daily_sales['date'] == sample_date)]
        
        print(f"示例：{sample_category} 在 {sample_date}")
        print(f"  单品数量: {len(original_data)}")
        print(f"  手动计算销售单价: {manual_price:.4f}")
        print(f"  程序计算销售单价: {calculated_data['price'].iloc[0]:.4f}")
        print(f"  手动计算批发价格: {manual_wholesale:.4f}")
        print(f"  程序计算批发价格: {calculated_data['wholesale_price'].iloc[0]:.4f}")
    
    return daily_sales, None  # 不再需要单独的wholesale_df

def create_sequences(data, look_back=30):
    """创建LSTM的输入序列"""
    
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, units=50, dropout=0.2):
    """构建LSTM模型"""
    
    model = Sequential([
        LSTM(units=units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(units=units, return_sequences=False),
        Dropout(dropout),
        Dense(units=25),
        Dense(units=1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def train_lstm_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """训练LSTM模型"""
    
    print(f"训练数据形状: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"验证数据形状: X_val={X_val.shape}, y_val={y_val.shape}")
    
    # 构建模型
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # 显示模型结构
    print("\n=== LSTM模型结构 ===")
    model.summary()
    
    # 早停机制
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    # 训练模型
    print(f"\n=== 开始训练 ===")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 显示训练结果
    print(f"\n=== 训练完成 ===")
    print(f"最终训练损失: {history.history['loss'][-1]:.6f}")
    print(f"最终验证损失: {history.history['val_loss'][-1]:.6f}")
    print(f"训练轮数: {len(history.history['loss'])}")
    
    return model, history

def plot_training_history(history, category_name):
    """绘制训练历史"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'{category_name} - LSTM训练历史', fontsize=16, fontweight='bold')
    
    # 损失曲线
    ax1 = axes[0]
    ax1.plot(history.history['loss'], 'b-', label='训练损失', linewidth=2)
    ax1.plot(history.history['val_loss'], 'r-', label='验证损失', linewidth=2)
    ax1.set_title('损失曲线')
    ax1.set_xlabel('训练轮数')
    ax1.set_ylabel('损失值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 损失曲线（对数尺度）
    ax2 = axes[1]
    ax2.semilogy(history.history['loss'], 'b-', label='训练损失', linewidth=2)
    ax2.semilogy(history.history['val_loss'], 'r-', label='验证损失', linewidth=2)
    ax2.set_title('损失曲线（对数尺度）')
    ax2.set_xlabel('训练轮数')
    ax2.set_ylabel('损失值（对数）')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{category_name}_训练历史.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_model_info(model, category_name, model_type):
    """打印模型详细信息"""
    
    print(f"\n=== {category_name} {model_type}模型信息 ===")
    print(f"模型参数总数: {model.count_params():,}")
    print(f"可训练参数: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    print(f"非可训练参数: {sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]):,}")
    
    # 显示每层信息
    print(f"\n层结构:")
    for i, layer in enumerate(model.layers):
        print(f"  第{i+1}层: {layer.name} - 输出形状: {layer.output_shape}")
        if hasattr(layer, 'units'):
            print(f"    单元数: {layer.units}")
        if hasattr(layer, 'rate'):
            print(f"    Dropout率: {layer.rate}")
    """验证LSTM模型的准确性，计算各种误差指标"""
    
    print(f"\n=== {category_name} LSTM模型验证 ===")
    
    try:
        # 预测
        y_pred = model.predict(X_test)
        
        # 反归一化
        y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_original = scaler.inverse_transform(y_pred).flatten()
        
        # 计算各种误差指标
        # 1. 均方误差 (MSE)
        mse = mean_squared_error(y_test_original, y_pred_original)
        
        # 2. 均方根误差 (RMSE)
        rmse = np.sqrt(mse)
        
        # 3. 平均绝对误差 (MAE)
        mae = mean_absolute_error(y_test_original, y_pred_original)
        
        # 4. 平均绝对百分比误差 (MAPE)
        # 避免除零错误
        non_zero_mask = y_test_original != 0
        if np.sum(non_zero_mask) > 0:
            mape = np.mean(np.abs((y_test_original[non_zero_mask] - y_pred_original[non_zero_mask]) / y_test_original[non_zero_mask])) * 100
        else:
            mape = np.nan
        
        # 5. 决定系数 (R²)
        r_squared = r2_score(y_test_original, y_pred_original)
        
        # 6. 调整后的决定系数 (Adjusted R²)
        n = len(y_test_original)
        k = model.count_params()  # 模型参数个数
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1) if n > k + 1 else np.nan
        
        # 打印误差指标
        print(f"误差指标:")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  MAPE: {mape:.2f}%" if not np.isnan(mape) else "  MAPE: 无法计算")
        print(f"  R²: {r_squared:.4f}")
        print(f"  Adjusted R²: {adj_r_squared:.4f}" if not np.isnan(adj_r_squared) else "  Adjusted R²: 无法计算")
        
        # 返回验证结果
        validation_results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'y_test': y_test_original,
            'y_pred': y_pred_original
        }
        
        return validation_results
        
    except Exception as e:
        print(f"模型验证失败: {e}")
        return None

def plot_model_validation(validation_results, category_name):
    """绘制模型验证图表"""
    
    if validation_results is None:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{category_name} - LSTM 模型验证', fontsize=16, fontweight='bold')
    
    y_test = validation_results['y_test']
    y_pred = validation_results['y_pred']
    
    # 1. 实际值 vs 预测值
    ax1 = axes[0, 0]
    ax1.plot(y_test, 'b-', label='实际值', linewidth=1.5)
    ax1.plot(y_pred, 'r--', label='预测值', linewidth=1.5)
    ax1.set_title('实际值 vs 预测值')
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 散点图
    ax2 = axes[0, 1]
    ax2.scatter(y_test, y_pred, alpha=0.6)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_title('实际值 vs 预测值散点图')
    ax2.set_xlabel('实际值')
    ax2.set_ylabel('预测值')
    ax2.grid(True, alpha=0.3)
    
    # 3. 残差图
    ax3 = axes[1, 0]
    residuals = y_test - y_pred
    ax3.plot(residuals, 'g-', linewidth=1)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax3.set_title('残差图')
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('残差')
    ax3.grid(True, alpha=0.3)
    
    # 4. 残差直方图
    ax4 = axes[1, 1]
    ax4.hist(residuals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_title('残差分布直方图')
    ax4.set_xlabel('残差')
    ax4.set_ylabel('频数')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{category_name}_LSTM模型验证.png', dpi=300, bbox_inches='tight')
    plt.close()

def forecast_demand(daily_sales, loss_df, forecast_dates, look_back=30):
    """使用LSTM预测需求"""
    
    print(f"\n=== 开始使用LSTM预测 {forecast_dates[0]} 到 {forecast_dates[-1]} 的需求 ===")
    
    categories = daily_sales['category'].unique()
    predictions = {}
    
    # 计算需求参数（价格-销量关系）
    demand_params = {}
    for cat in categories:
        cat_data = daily_sales[daily_sales['category'] == cat]
        if len(cat_data) > 10:  # 确保有足够的数据
            X = cat_data['price'].values.reshape(-1, 1)
            y = cat_data['quantity'].values
            
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression().fit(X, y)
            a, b = reg.intercept_, -reg.coef_[0]  # Q = a - b * P
            
            demand_params[cat] = {'a': a, 'b': b}
            
            # 计算价格弹性
            mean_P = cat_data['price'].mean()
            mean_Q = cat_data['quantity'].mean()
            elasticity = -b * (mean_P / mean_Q)
            
            print(f"\n{cat}:")
            print(f"  需求函数: Q = {a:.2f} - {b:.4f} * P")
            print(f"  价格弹性: {elasticity:.4f}")
    
    # 对每个品类进行预测
    for cat in categories:
        print(f"\n--- 预测 {cat} ---")
        
        # 获取该品类的销量时间序列
        cat_sales = daily_sales[daily_sales['category'] == cat].copy()
        cat_sales = cat_sales.sort_values('date').set_index('date')
        
        if len(cat_sales) < look_back + 20:  # 数据太少，跳过
            print(f"  {cat} 数据量不足，跳过预测")
            continue
        
        # 销量预测
        ts_quantity = cat_sales['quantity'].values
        
        # 数据标准化
        scaler_q = MinMaxScaler()
        ts_quantity_scaled = scaler_q.fit_transform(ts_quantity.reshape(-1, 1))
        
        # 创建序列
        X_q, y_q = create_sequences(ts_quantity_scaled, look_back)
        
        # 划分训练集和验证集
        split_idx = int(len(X_q) * 0.8)
        X_train_q, X_val_q = X_q[:split_idx], X_q[split_idx:]
        y_train_q, y_val_q = y_q[:split_idx], y_q[split_idx:]
        
        # 训练销量LSTM模型
        print(f"    正在训练销量LSTM模型...")
        model_q, history_q = train_lstm_model(X_train_q, y_train_q, X_val_q, y_val_q)
        
        # 绘制训练历史
        plot_training_history(history_q, f"{cat}_销量")
        
        # 打印模型信息
        print_model_info(model_q, cat, "销量")
        
        # 验证销量模型
        validation_results_q = validate_lstm_model(model_q, X_val_q, y_val_q, scaler_q, f"{cat}_销量")
        
        # 绘制模型验证图表
        if validation_results_q is not None:
            plot_model_validation(validation_results_q, f"{cat}_销量")
        
        # 预测销量
        last_sequence = ts_quantity_scaled[-look_back:].reshape(1, look_back, 1)
        q_forecast = []
        
        for _ in range(len(forecast_dates)):
            next_pred = model_q.predict(last_sequence)
            q_forecast.append(next_pred[0, 0])
            
            # 更新序列（移除最早的值，添加新预测值）
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = next_pred[0, 0]
        
        # 反归一化
        q_forecast = scaler_q.inverse_transform(np.array(q_forecast).reshape(-1, 1)).flatten()
        
        print(f"    销量预测: {q_forecast.mean():.2f} ± {q_forecast.std():.2f}")
        
        # 批发价格预测
        ts_wholesale = cat_sales['wholesale_price'].values
        
        # 数据标准化
        scaler_c = MinMaxScaler()
        ts_wholesale_scaled = scaler_c.fit_transform(ts_wholesale.reshape(-1, 1))
        
        # 创建序列
        X_c, y_c = create_sequences(ts_wholesale_scaled, look_back)
        
        if len(X_c) > 10:  # 确保有足够的数据
            # 划分训练集和验证集
            split_idx = int(len(X_c) * 0.8)
            X_train_c, X_val_c = X_c[:split_idx], X_c[split_idx:]
            y_train_c, y_val_c = y_c[:split_idx], y_c[split_idx:]
            
            # 训练批发价格LSTM模型
            print(f"    正在训练批发价格LSTM模型...")
            model_c, history_c = train_lstm_model(X_train_c, y_train_c, X_val_c, y_val_c)
            
            # 绘制训练历史
            plot_training_history(history_c, f"{cat}_批发价格")
            
            # 打印模型信息
            print_model_info(model_c, cat, "批发价格")
            
            # 验证批发价格模型
            validation_results_c = validate_lstm_model(model_c, X_val_c, y_val_c, scaler_c, f"{cat}_批发价格")
            
            # 预测批发价格
            last_sequence_c = ts_wholesale_scaled[-look_back:].reshape(1, look_back, 1)
            c_forecast = []
            
            for _ in range(len(forecast_dates)):
                next_pred = model_c.predict(last_sequence_c)
                c_forecast.append(next_pred[0, 0])
                
                # 更新序列
                last_sequence_c = np.roll(last_sequence_c, -1, axis=1)
                last_sequence_c[0, -1, 0] = next_pred[0, 0]
            
            # 反归一化
            c_forecast = scaler_c.inverse_transform(np.array(c_forecast).reshape(-1, 1)).flatten()
            
            print(f"    批发价格预测: {c_forecast.mean():.2f} ± {c_forecast.std():.2f}")
        else:
            # 数据太少，使用移动平均
            print(f"    {cat} 批发价格数据量不足，使用移动平均")
            c_forecast = cat_sales['wholesale_price'].rolling(window=7, min_periods=1).mean().iloc[-1]
            c_forecast = [c_forecast] * len(forecast_dates)
        
        # 获取损耗率
        if '分类名称' in loss_df.columns:
            cat_loss = (loss_df[loss_df['分类名称'] == cat]['损耗率(%)'].mean())/100
        else:
            cat_loss = 0.05  # 默认5%损耗率
        
        predictions[cat] = {
            'q_forecast': q_forecast,
            'c_forecast': c_forecast,
            'loss_rate': cat_loss,
            'demand_params': demand_params.get(cat, {}),
            'lstm_model_q': model_q,
            'lstm_model_c': model_c if 'model_c' in locals() else None,
            'validation_results_q': validation_results_q,
            'validation_results_c': validation_results_c if 'validation_results_c' in locals() else None
        }
        
        print(f"  {cat} 预测完成:")
        print(f"    销量预测: {q_forecast.mean():.2f} ± {q_forecast.std():.2f}")
        print(f"    批发价格预测: {np.mean(c_forecast):.2f}")
        print(f"    损耗率: {cat_loss:.2%}")
    
    return predictions

def create_forecast_visualizations(daily_sales, predictions, forecast_dates):
    """创建预测结果可视化"""
    
    print("\n=== 创建预测结果可视化 ===")
    
    categories = list(predictions.keys())
    n_categories = len(categories)
    
    if n_categories == 0:
        print("没有可用的预测结果")
        return
    
    # 创建子图
    fig, axes = plt.subplots(n_categories, 1, figsize=(15, 5*n_categories))
    if n_categories == 1:
        axes = [axes]
    
    for i, cat in enumerate(categories):
        ax = axes[i]
        
        # 获取历史数据
        cat_data = daily_sales[daily_sales['category'] == cat].copy()
        cat_data = cat_data.sort_values('date').set_index('date')
        
        # 绘制历史销量
        ax.plot(cat_data.index, cat_data['quantity'], 'b-', label='历史销量', linewidth=1)
        
        # 绘制预测销量
        forecast_values = predictions[cat]['q_forecast']
        ax.plot(forecast_dates, forecast_values, 'r--', label='预测销量', linewidth=2)
        
        # 添加预测区间
        ax.fill_between(forecast_dates, 
                       forecast_values - forecast_values.std(),
                       forecast_values + forecast_values.std(),
                       alpha=0.3, color='red', label='预测区间')
        
        ax.set_title(f'{cat} - LSTM销量预测', fontsize=14, fontweight='bold')
        ax.set_xlabel('日期')
        ax.set_ylabel('销量(千克)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 旋转x轴标签
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('LSTM需求预测结果.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_wholesale_price_visualizations(daily_sales, predictions, forecast_dates):
    """创建批发价格预测结果可视化"""
    
    print("\n=== 创建批发价格预测结果可视化 ===")
    
    categories = list(predictions.keys())
    n_categories = len(categories)
    
    if n_categories == 0:
        print("没有可用的预测结果")
        return
    
    # 创建子图
    fig, axes = plt.subplots(n_categories, 1, figsize=(15, 5*n_categories))
    if n_categories == 1:
        axes = [axes]
    
    for i, cat in enumerate(categories):
        ax = axes[i]
        
        # 获取历史数据
        cat_data = daily_sales[daily_sales['category'] == cat].copy()
        cat_data = cat_data.sort_values('date').set_index('date')
        
        # 绘制历史批发价格
        ax.plot(cat_data.index, cat_data['wholesale_price'], 'g-', label='历史批发价格', linewidth=1)
        
        # 绘制预测批发价格
        forecast_values = predictions[cat]['c_forecast']
        ax.plot(forecast_dates, forecast_values, 'r--', label='预测批发价格', linewidth=2)
        
        # 添加预测区间
        ax.fill_between(forecast_dates, 
                       forecast_values - forecast_values.std(),
                       forecast_values + forecast_values.std(),
                       alpha=0.3, color='red', label='预测区间')
        
        ax.set_title(f'{cat} - LSTM批发价格预测', fontsize=14, fontweight='bold')
        ax.set_xlabel('日期')
        ax.set_ylabel('批发价格(元/千克)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 旋转x轴标签
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('LSTM批发价格预测结果.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_model_validation_summary(predictions):
    """保存模型验证结果汇总"""
    
    print("\n=== 保存LSTM模型验证结果汇总 ===")
    
    validation_summary = []
    for cat, pred in predictions.items():
        if pred.get('validation_results_q') is not None:
            val = pred['validation_results_q']
            validation_summary.append({
                '品类': cat,
                '预测类型': '销量',
                'MSE': val['mse'],
                'RMSE': val['rmse'],
                'MAE': val['mae'],
                'MAPE(%)': val['mape'] if not np.isnan(val['mape']) else np.nan,
                'R²': val['r_squared'],
                'Adjusted_R²': val['adj_r_squared'] if not np.isnan(val['adj_r_squared']) else np.nan
            })
        
        if pred.get('validation_results_c') is not None:
            val = pred['validation_results_c']
            validation_summary.append({
                '品类': cat,
                '预测类型': '批发价格',
                'MSE': val['mse'],
                'RMSE': val['rmse'],
                'MAE': val['mae'],
                'MAPE(%)': val['mape'] if not np.isnan(val['mape']) else np.nan,
                'R²': val['r_squared'],
                'Adjusted_R²': val['adj_r_squared'] if not np.isnan(val['adj_r_squared']) else np.nan
            })
    
    if validation_summary:
        validation_df = pd.DataFrame(validation_summary)
        validation_df.to_csv('LSTM模型验证结果汇总.csv', index=False, encoding='utf-8-sig')
        
        print("LSTM模型验证结果已保存到 'LSTM模型验证结果汇总.csv'")
        
        # 打印汇总统计
        print("\n=== LSTM模型验证结果汇总 ===")
        print(validation_df.round(4))
        
        # 计算平均误差指标
        print("\n=== 平均误差指标 ===")
        numeric_cols = ['MSE', 'RMSE', 'MAE', 'R²']
        avg_metrics = validation_df[numeric_cols].mean()
        print(avg_metrics.round(4))
        
        return validation_df
    else:
        print("没有可用的验证结果")
        return None

def save_predictions(predictions, forecast_dates):
    """保存预测结果"""
    
    print("\n=== 保存预测结果 ===")
    
    # 创建预测结果DataFrame
    results = []
    for cat, pred in predictions.items():
        print(f"\n处理品类: {cat}")
        
        # 确保预测结果是正确的格式
        q_forecast = pred['q_forecast']
        c_forecast = pred['c_forecast']
        
        # 确保是列表格式
        if not isinstance(q_forecast, (list, np.ndarray)):
            print(f"警告: {cat} 的销量预测格式不正确，跳过")
            continue
            
        if not isinstance(c_forecast, (list, np.ndarray)):
            print(f"警告: {cat} 的批发价格预测格式不正确，跳过")
            continue
        
        # 检查长度是否匹配
        if len(q_forecast) != len(forecast_dates):
            print(f"警告: {cat} 预测长度 {len(q_forecast)} 与日期长度 {len(forecast_dates)} 不匹配")
            continue
        
        # 创建结果记录
        for i, date in enumerate(forecast_dates):
            try:
                results.append({
                    '日期': date,
                    '品类': cat,
                    '预测销量(千克)': float(q_forecast[i]),
                    '预测批发价格(元/千克)': float(c_forecast[i]),
                    '损耗率': float(pred['loss_rate']),
                    '建议进货量(千克)': float(q_forecast[i]) / (1 - float(pred['loss_rate']))
                })
            except (IndexError, TypeError, ValueError) as e:
                print(f"错误: 处理 {cat} 第 {i} 个预测值时出错: {e}")
                continue
    
    if not results:
        print("没有有效的预测结果可以保存")
        return None
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('LSTM需求预测结果.csv', index=False, encoding='utf-8-sig')
    
    print("预测结果已保存到 'LSTM需求预测结果.csv'")
    
    # 打印汇总信息
    print("\n=== 预测结果汇总 ===")
    summary = results_df.groupby('品类').agg({
        '预测销量(千克)': ['mean', 'sum'],
        '建议进货量(千克)': 'sum'
    }).round(2)
    
    print(summary)
    
    return results_df

def main():
    """主函数"""
    
    try:
        # 加载数据
        sales_df, wholesale_df, loss_df = load_and_preprocess_data()
        if sales_df is None:
            return
        
        # 准备每日数据
        daily_sales, _ = prepare_daily_data(sales_df, wholesale_df)
        
        # 定义预测日期范围
        forecast_dates = pd.date_range('2023-07-01', '2023-07-07', freq='D')
        
        # 预测需求
        predictions = forecast_demand(daily_sales, loss_df, forecast_dates)
        
        if predictions:
            # 创建可视化
            create_forecast_visualizations(daily_sales, predictions, forecast_dates)
            create_wholesale_price_visualizations(daily_sales, predictions, forecast_dates)
            
            # 保存预测结果
            results_df = save_predictions(predictions, forecast_dates)
            
            # 保存模型验证结果
            validation_df = save_model_validation_summary(predictions)
            
            if results_df is not None:
                print("\n=== LSTM需求预测完成 ===")
                print("输出文件:")
                print("- LSTM需求预测结果.csv")
                print("- LSTM需求预测结果.png")
                print("- LSTM批发价格预测结果.png")
                print("- 各品类LSTM模型验证图")
                print("- 各品类LSTM训练历史图")
                print("- LSTM模型验证结果汇总.csv")
            else:
                print("预测结果保存失败")
        else:
            print("没有成功生成预测结果")
            
    except Exception as e:
        print(f"预测过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
