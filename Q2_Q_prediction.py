# 用ARIMA预测'2023-07-01', '2023-07-07'期间的需求

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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
    
    # 加载损耗率数据（使用附件4.xlsx）
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

def check_stationarity(ts_data, title):
    """检查时间序列的平稳性"""
    
    print(f"\n=== {title} 平稳性检验 ===")
    
    # ADF检验
    result = adfuller(ts_data.dropna())
    print(f'ADF统计量: {result[0]:.4f}')
    print(f'p值: {result[1]:.4f}')
    
    if result[1] <= 0.05:
        print("时间序列是平稳的 (p <= 0.05)")
        return True
    else:
        print("时间序列不是平稳的 (p > 0.05)")
        return False

def make_stationary(ts_data, max_diff=3):
    """通过差分使时间序列平稳，返回差分后的序列和差分阶数"""
    
    print(f"正在通过差分使时间序列平稳...")
    
    current_series = ts_data.copy()
    diff_order = 0
    
    for d in range(max_diff + 1):
        if d == 0:
            # 原始序列
            is_stationary = check_stationarity(current_series, f"{d}阶差分")
            if is_stationary:
                print(f"原始序列已经是平稳的，无需差分")
                return current_series, d
        else:
            # 进行d阶差分
            current_series = current_series.diff().dropna()
            diff_order = d
            
            if len(current_series) < 10:  # 差分后数据太少
                print(f"差分后数据量不足，回退到{d-1}阶差分")
                # 回退到上一阶差分
                current_series = ts_data.copy()
                for i in range(d-1):
                    current_series = current_series.diff().dropna()
                return current_series, d-1
            
            is_stationary = check_stationarity(current_series, f"{d}阶差分")
            if is_stationary:
                print(f"经过{d}阶差分后，时间序列变为平稳")
                return current_series, d
    
    print(f"经过{max_diff}阶差分后仍不平稳，使用{max_diff}阶差分")
    return current_series, max_diff

def find_best_arima_order(ts_data, d_order, max_p=7, max_q=7):
    """寻找最佳ARIMA参数，d_order是已经确定的差分阶数"""
    
    print(f"正在寻找最佳ARIMA参数 (d={d_order})...")
    
    best_aic = np.inf
    best_order = None
    
    # 尝试不同的p和q参数组合，d已经确定
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                model = ARIMA(ts_data, order=(p, d_order, q))
                fitted_model = model.fit()
                aic = fitted_model.aic
                
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d_order, q)
                    
            except:
                continue
    
    print(f"最佳ARIMA参数: {best_order}, AIC: {best_aic:.2f}")
    return best_order

def fit_arima_model(ts_data, order, seasonal_order=None):
    """拟合ARIMA模型"""
    
    try:
        if seasonal_order:
            model = ARIMA(ts_data, order=order, seasonal_order=seasonal_order)
        else:
            model = ARIMA(ts_data, order=order)
        
        fitted_model = model.fit()
        return fitted_model
    except Exception as e:
        print(f"ARIMA模型拟合失败: {e}")
        return None

def validate_arima_model(model, ts_data, order, category_name):
    """验证ARIMA模型的准确性，计算各种误差指标"""
    
    print(f"\n=== {category_name} ARIMA模型验证 ===")
    
    try:
        # 获取模型参数
        p, d, q = order
        print(f"模型参数: ARIMA({p},{d},{q})")
        
        # 使用模型对历史数据进行拟合值计算
        fitted_values = model.fittedvalues
        
        # 计算残差
        residuals = model.resid
        
        # 计算各种误差指标
        # 1. 均方误差 (MSE)
        mse = np.mean(residuals**2)
        
        # 2. 均方根误差 (RMSE)
        rmse = np.sqrt(mse)
        
        # 3. 平均绝对误差 (MAE)
        mae = np.mean(np.abs(residuals))
        
        # 4. 平均绝对百分比误差 (MAPE)
        # 避免除零错误
        non_zero_data = ts_data[ts_data != 0]
        if len(non_zero_data) > 0:
            mape = np.mean(np.abs(residuals[ts_data != 0] / non_zero_data)) * 100
        else:
            mape = np.nan
        
        # 5. 决定系数 (R²)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((ts_data - ts_data.mean())**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        
        # 6. 调整后的决定系数 (Adjusted R²)
        n = len(ts_data)
        k = p + q  # 参数个数
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1) if n > k + 1 else np.nan
        
        # 7. AIC和BIC
        aic = model.aic
        bic = model.bic
        
        # 8. Ljung-Box检验残差的白噪声性
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_test = acorr_ljungbox(residuals, lags=min(10, len(residuals)//5), return_df=True)
            lb_pvalue = lb_test['lb_pvalue'].iloc[-1]
            lb_result = "残差是白噪声" if lb_pvalue > 0.05 else "残差不是白噪声"
        except:
            lb_pvalue = np.nan
            lb_result = "无法计算"
        
        # 打印误差指标
        print(f"误差指标:")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  MAPE: {mape:.2f}%" if not np.isnan(mape) else "  MAPE: 无法计算")
        print(f"  R²: {r_squared:.4f}")
        print(f"  Adjusted R²: {adj_r_squared:.4f}" if not np.isnan(adj_r_squared) else "  Adjusted R²: 无法计算")
        print(f"  AIC: {aic:.2f}")
        print(f"  BIC: {bic:.2f}")
        print(f"  Ljung-Box检验: {lb_result} (p={lb_pvalue:.4f})")
        
        # 返回验证结果
        validation_results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'aic': aic,
            'bic': bic,
            'ljung_box_pvalue': lb_pvalue,
            'fitted_values': fitted_values,
            'residuals': residuals
        }
        
        return validation_results
        
    except Exception as e:
        print(f"模型验证失败: {e}")
        return None

def plot_model_validation(ts_data, validation_results, category_name, order):
    """绘制模型验证图表"""
    
    if validation_results is None:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{category_name} - ARIMA{order} 模型验证', fontsize=16, fontweight='bold')
    
    # 1. 实际值 vs 拟合值
    ax1 = axes[0, 0]
    fitted_values = validation_results['fitted_values']
    ax1.plot(ts_data.index, ts_data.values, 'b-', label='实际值', linewidth=1.5)
    ax1.plot(ts_data.index, fitted_values, 'r--', label='拟合值', linewidth=1.5)
    ax1.set_title('实际值 vs 拟合值')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('销量(千克)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 残差图
    ax2 = axes[0, 1]
    residuals = validation_results['residuals']
    ax2.plot(ts_data.index, residuals, 'g-', linewidth=1)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax2.set_title('残差图')
    ax2.set_xlabel('日期')
    ax2.set_ylabel('残差')
    ax2.grid(True, alpha=0.3)
    
    # 3. 残差直方图
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_title('残差分布直方图')
    ax3.set_xlabel('残差')
    ax3.set_ylabel('频数')
    ax3.grid(True, alpha=0.3)
    
    # 4. Q-Q图（正态性检验）
    ax4 = axes[1, 1]
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('残差Q-Q图（正态性检验）')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{category_name}_模型验证.png', dpi=300, bbox_inches='tight')
    plt.close()

def forecast_demand(daily_sales, loss_df, forecast_dates):
    """预测需求
    
    按照Q2.md中的公式计算加权平均价格：
    P_i(t) = Σ(p_j(t) * q_j(t)) / Σ(q_j(t))  # 加权平均销售单价
    C_i(t) = Σ(c_j(t) * q_j(t)) / Σ(q_j(t))  # 加权平均批发价格
    
    其中：
    - p_j(t) 是单品j在时间t的销售单价
    - q_j(t) 是单品j在时间t的销量
    - c_j(t) 是单品j在时间t的批发价格
    """
    
    print(f"\n=== 开始预测 {forecast_dates[0]} 到 {forecast_dates[-1]} 的需求 ===")
    
    categories = daily_sales['category'].unique()
    predictions = {}
    
    # 计算需求参数（价格-销量关系）
    demand_params = {}
    for cat in categories:
        cat_data = daily_sales[daily_sales['category'] == cat]
        if len(cat_data) > 10:  # 确保有足够的数据
            X = cat_data['price'].values.reshape(-1, 1)
            y = cat_data['quantity'].values
            
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
        
        if len(cat_sales) < 20:  # 数据太少，跳过
            print(f"  {cat} 数据量不足，跳过预测")
            continue
        
        # 销量预测
        ts_quantity = cat_sales['quantity']
        
        # 通过平稳性检验确定差分阶数
        stationary_series, d_order = make_stationary(ts_quantity, max_diff=3)
        
        # 可视化差分过程
        plot_differencing_process(ts_quantity, d_order, cat)
        
        # 寻找最佳ARIMA参数（d已经确定）
        best_order = find_best_arima_order(stationary_series, d_order, max_p=7, max_q=7)
        
        # 拟合ARIMA模型（使用原始序列和确定的差分阶数）
        model_q = fit_arima_model(ts_quantity, best_order)
        
        if model_q is None:
            print(f"  {cat} ARIMA模型拟合失败")
            continue
        
        # 验证ARIMA模型的准确性
        validation_results = validate_arima_model(model_q, ts_quantity, best_order, cat)
        
        # 绘制模型验证图表
        if validation_results is not None:
            plot_model_validation(ts_quantity, validation_results, cat, best_order)
        
        # 预测销量
        q_forecast = model_q.forecast(steps=len(forecast_dates))
        
        # 确保预测结果是正确的格式
        if hasattr(q_forecast, 'values'):
            q_forecast = q_forecast.values
        elif hasattr(q_forecast, 'tolist'):
            q_forecast = np.array(q_forecast)
        
        print(f"    销量预测类型: {type(q_forecast)}")
        print(f"    销量预测长度: {len(q_forecast)}")
        print(f"    销量预测前3个值: {q_forecast[:3] if len(q_forecast) >= 3 else q_forecast}")
        
        # 批发价格预测（使用ARIMA模型）
        cat_wholesale_prices = cat_sales['wholesale_price']
        if len(cat_wholesale_prices) > 20:  # 确保有足够的数据进行ARIMA建模
            print(f"    正在为 {cat} 建立批发价格ARIMA模型...")
            
            # 通过平稳性检验确定差分阶数
            stationary_wholesale, d_order_wholesale = make_stationary(cat_wholesale_prices, max_diff=3)
            
            # 寻找最佳ARIMA参数（d已经确定）
            best_order_wholesale = find_best_arima_order(stationary_wholesale, d_order_wholesale, max_p=7, max_q=7)
            
            # 拟合ARIMA模型
            model_c = fit_arima_model(cat_wholesale_prices, best_order_wholesale)
            
            if model_c is not None:
                # 预测批发价格
                c_forecast = model_c.forecast(steps=len(forecast_dates))
                
                # 确保预测结果是正确的格式
                if hasattr(c_forecast, 'values'):
                    c_forecast = c_forecast.values
                elif hasattr(c_forecast, 'tolist'):
                    c_forecast = np.array(c_forecast)
                
                print(f"    批发价格ARIMA模型: ARIMA{best_order_wholesale}")
                print(f"    批发价格预测: {c_forecast.mean():.2f} ± {c_forecast.std():.2f}")
            else:
                # ARIMA模型拟合失败，使用移动平均作为备选
                print(f"    批发价格ARIMA模型拟合失败，使用移动平均作为备选")
                c_forecast = cat_wholesale_prices.rolling(window=7, min_periods=1).mean().iloc[-1]
                c_forecast = [c_forecast] * len(forecast_dates)
        else:
            # 数据太少，使用移动平均
            print(f"    {cat} 批发价格数据量不足，使用移动平均")
            if len(cat_wholesale_prices) > 0:
                c_forecast = cat_wholesale_prices.rolling(window=7, min_periods=1).mean().iloc[-1]
                c_forecast = [c_forecast] * len(forecast_dates)
            else:
                # 如果没有批发价格数据，使用销售价格的移动平均
                c_forecast = cat_sales['price'].rolling(window=7, min_periods=1).mean().iloc[-1] * 0.8
                c_forecast = [c_forecast] * len(forecast_dates)
        
        print(f"    批发价格预测类型: {type(c_forecast)}")
        print(f"    批发价格预测长度: {len(c_forecast)}")
        print(f"    批发价格预测前3个值: {c_forecast[:3] if len(c_forecast) >= 3 else c_forecast}")
        
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
            'arima_model': model_q,
            'wholesale_arima_model': model_c if 'model_c' in locals() else None,
            'wholesale_arima_order': best_order_wholesale if 'best_order_wholesale' in locals() else None,
            'validation_results': validation_results
        }
        
        print(f"  {cat} 预测完成:")
        print(f"    销量预测: {q_forecast.mean():.2f} ± {q_forecast.std():.2f}")
        print(f"    批发价格预测: {np.mean(c_forecast):.2f}")
        print(f"    损耗率: {cat_loss:.2%}")
    
    return predictions

def plot_differencing_process(ts_data, d_order, category_name):
    """可视化差分过程"""
    
    plt.figure(figsize=(15, 10))
    
    # 原始序列
    plt.subplot(d_order + 2, 1, 1)
    plt.plot(ts_data.index, ts_data.values, 'b-', linewidth=1.5)
    plt.title(f'{category_name} - 原始销量序列', fontsize=12, fontweight='bold')
    plt.ylabel('销量(千克)')
    plt.grid(True, alpha=0.3)
    
    # 各阶差分序列
    current_series = ts_data.copy()
    for i in range(1, d_order + 1):
        current_series = current_series.diff().dropna()
        plt.subplot(d_order + 2, 1, i + 1)
        plt.plot(current_series.index, current_series.values, 'g-', linewidth=1.5)
        plt.title(f'{i}阶差分序列', fontsize=12, fontweight='bold')
        plt.ylabel(f'Δ^{i}销量')
        plt.grid(True, alpha=0.3)
    
    # 平稳序列（最终差分后的序列）
    plt.subplot(d_order + 2, 1, d_order + 2)
    plt.plot(current_series.index, current_series.values, 'r-', linewidth=1.5)
    plt.title(f'{d_order}阶差分后的平稳序列', fontsize=12, fontweight='bold')
    plt.ylabel(f'Δ^{d_order}销量')
    plt.xlabel('日期')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{category_name}_差分过程.png', dpi=300, bbox_inches='tight')
    plt.close()

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
        
        ax.set_title(f'{cat} - 销量预测', fontsize=14, fontweight='bold')
        ax.set_xlabel('日期')
        ax.set_ylabel('销量(千克)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 旋转x轴标签
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('需求预测结果.png', dpi=300, bbox_inches='tight')
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
        
        ax.set_title(f'{cat} - 批发价格预测', fontsize=14, fontweight='bold')
        ax.set_xlabel('日期')
        ax.set_ylabel('批发价格(元/千克)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 旋转x轴标签
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('批发价格预测结果.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_model_validation_summary(predictions):
    """保存模型验证结果汇总"""
    
    print("\n=== 保存模型验证结果汇总 ===")
    
    validation_summary = []
    for cat, pred in predictions.items():
        if pred.get('validation_results') is not None:
            val = pred['validation_results']
            validation_summary.append({
                '品类': cat,
                'MSE': val['mse'],
                'RMSE': val['rmse'],
                'MAE': val['mae'],
                'MAPE(%)': val['mape'] if not np.isnan(val['mape']) else np.nan,
                'R²': val['r_squared'],
                'Adjusted_R²': val['adj_r_squared'] if not np.isnan(val['adj_r_squared']) else np.nan,
                'AIC': val['aic'],
                'BIC': val['bic'],
                'Ljung_Box_p值': val['ljung_box_pvalue'] if not np.isnan(val['ljung_box_pvalue']) else np.nan
            })
    
    if validation_summary:
        validation_df = pd.DataFrame(validation_summary)
        validation_df.to_csv('ARIMA模型验证结果汇总.csv', index=False, encoding='utf-8-sig')
        
        print("模型验证结果已保存到 'ARIMA模型验证结果汇总.csv'")
        
        # 打印汇总统计
        print("\n=== 模型验证结果汇总 ===")
        print(validation_df.round(4))
        
        # 计算平均误差指标
        print("\n=== 平均误差指标 ===")
        numeric_cols = ['MSE', 'RMSE', 'MAE', 'R²', 'AIC', 'BIC']
        avg_metrics = validation_df[numeric_cols].mean()
        print(avg_metrics.round(4))
        
        return validation_df
    else:
        print("没有可用的验证结果")
        return None

def save_wholesale_price_validation_summary(predictions):
    """保存批发价格ARIMA模型验证结果汇总"""
    
    print("\n=== 保存批发价格ARIMA模型验证结果汇总 ===")
    
    validation_summary = []
    for cat, pred in predictions.items():
        if pred.get('wholesale_arima_model') is not None:
            model = pred['wholesale_arima_model']
            order = pred.get('wholesale_arima_order', 'Unknown')
            
            # 获取历史批发价格数据用于验证
            # 这里需要从daily_sales中获取，暂时跳过详细验证
            validation_summary.append({
                '品类': cat,
                'ARIMA参数': str(order),
                'AIC': model.aic,
                'BIC': model.bic,
                '模型状态': '已拟合'
            })
    
    if validation_summary:
        validation_df = pd.DataFrame(validation_summary)
        validation_df.to_csv('批发价格ARIMA模型验证结果汇总.csv', index=False, encoding='utf-8-sig')
        
        print("批发价格ARIMA模型验证结果已保存到 '批发价格ARIMA模型验证结果汇总.csv'")
        
        # 打印汇总统计
        print("\n=== 批发价格ARIMA模型验证结果汇总 ===")
        print(validation_df.round(4))
        
        return validation_df
    else:
        print("没有可用的批发价格ARIMA模型验证结果")
        return None

def save_predictions(predictions, forecast_dates):
    """保存预测结果"""
    
    print("\n=== 保存预测结果 ===")
    
    # 创建预测结果DataFrame
    results = []
    for cat, pred in predictions.items():
        print(f"\n处理品类: {cat}")
        print(f"预测结果类型: {type(pred['q_forecast'])}")
        print(f"预测结果长度: {len(pred['q_forecast']) if hasattr(pred['q_forecast'], '__len__') else '无长度属性'}")
        
        # 确保预测结果是正确的格式
        q_forecast = pred['q_forecast']
        c_forecast = pred['c_forecast']
        
        # 如果是pandas Series，转换为列表
        if hasattr(q_forecast, 'values'):
            q_forecast = q_forecast.values.tolist()
        elif hasattr(q_forecast, 'tolist'):
            q_forecast = q_forecast.tolist()
        
        if hasattr(c_forecast, 'values'):
            c_forecast = c_forecast.values.tolist()
        elif hasattr(c_forecast, 'tolist'):
            c_forecast = c_forecast.tolist()
        
        # 确保是列表格式
        if not isinstance(q_forecast, list):
            print(f"警告: {cat} 的销量预测不是列表格式，跳过")
            continue
            
        if not isinstance(c_forecast, list):
            print(f"警告: {cat} 的批发价格预测不是列表格式，跳过")
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
    results_df.to_csv('需求预测结果.csv', index=False, encoding='utf-8-sig')
    
    print("预测结果已保存到 '需求预测结果.csv'")
    
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
            wholesale_validation_df = save_wholesale_price_validation_summary(predictions)
            
            if results_df is not None:
                print("\n=== 需求预测完成 ===")
                print("输出文件:")
                print("- 需求预测结果.csv")
                print("- 需求预测结果.png")
                print("- 批发价格预测结果.png")
                print("- 各品类差分过程图（用于平稳性检验）")
                print("- 各品类模型验证图")
                print("- ARIMA模型验证结果汇总.csv")
                print("- 批发价格ARIMA模型验证结果汇总.csv")
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
