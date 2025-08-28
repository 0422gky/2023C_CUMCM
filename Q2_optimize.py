# -*- coding: utf-8 -*-
"""
Q2步骤3：最优化模型（最大化收益）
基于线性回归的需求函数 Q = a_i - b_i * P
优化目标：最大化一周总收益
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_regression_results():
    """加载线性回归和log-log回归结果，获取需求函数参数"""
    try:
        # 加载线性回归结果
        linear_results = pd.read_csv('linear_regression_results.csv')
        print("成功加载线性回归结果")
        print(linear_results[['分类名称', 'intercept', 'slope', 'r2']].to_string(index=False))
        
        # 加载log-log回归结果
        loglog_results = pd.read_csv('loglog_regression_results.csv')
        print("\n成功加载Log-Log回归结果")
        print(loglog_results[['分类名称', 'intercept', 'log_P_coef', 'price_elasticity', 'r2']].to_string(index=False))
        
        return linear_results, loglog_results
    except FileNotFoundError as e:
        print(f"错误：找不到回归结果文件 - {e}")
        print("请先运行 Q2_regression_qp.py 生成回归结果")
        return None, None

def load_historical_data():
    """加载历史数据，用于计算历史平均价格和需求预测"""
    try:
        # 加载历史P-Q数据
        pq_data = pd.read_csv('file_23_pq.csv')
        pq_data['销售日期'] = pd.to_datetime(pq_data['销售日期'])
        
        # 计算每个分类的历史平均价格和销量
        historical_stats = pq_data.groupby('分类名称').agg({
            'P': ['mean', 'std'],
            'Q': ['mean', 'std']
        }).round(4)
        
        historical_stats.columns = ['P_avg', 'P_std', 'Q_avg', 'Q_std']
        historical_stats = historical_stats.reset_index()
        
        print("成功加载历史数据统计")
        print(historical_stats.to_string(index=False))
        
        return historical_stats, pq_data
        
    except FileNotFoundError:
        print("错误：找不到 file_23_pq.csv 文件")
        print("请先运行 Q2_regression_qp.py 生成P-Q数据")
        return None, None

def load_forecast_data():
    """加载需求预测结果.csv中的预测数据"""
    try:
        forecast_data = pd.read_csv('需求预测结果.csv')
        forecast_data['日期'] = pd.to_datetime(forecast_data['日期'])
        
        # 按分类和日期组织数据
        forecast_dict = {}
        for _, row in forecast_data.iterrows():
            category = row['品类']
            date = row['日期']
            
            if category not in forecast_dict:
                forecast_dict[category] = {}
            
            forecast_dict[category][date] = {
                'demand': row['预测销量(千克)'],
                'wholesale_price': row['预测批发价格(元/千克)'],
                'loss_rate': row['损耗率']
            }
        
        print("成功加载需求预测数据")
        print(f"预测期间: {forecast_data['日期'].min()} 到 {forecast_data['日期'].max()}")
        print(f"包含分类: {', '.join(forecast_data['品类'].unique())}")
        
        return forecast_dict, forecast_data
        
    except FileNotFoundError:
        print("错误：找不到 需求预测结果.csv 文件")
        return None, None

def get_forecast_for_date(forecast_dict, category, date):
    """获取指定分类和日期的预测数据"""
    if category in forecast_dict and date in forecast_dict[category]:
        return forecast_dict[category][date]
    else:
        print(f"警告：找不到 {category} 在 {date.strftime('%Y-%m-%d')} 的预测数据")
        return None

def estimate_loss_rates():
    """从file_14.csv加载各分类的真实损耗率数据"""
    try:
        # 读取损耗率数据
        loss_data = pd.read_csv('file_14.csv')
        
        # 计算每个分类的平均损耗率
        category_loss = loss_data.groupby('分类名称')['损耗率(%)'].mean()
        
        # 转换为小数形式（除以100）
        loss_rates = {}
        for category, loss_rate in category_loss.items():
            loss_rates[category] = loss_rate / 100
        
        print("从file_14.csv加载的真实损耗率：")
        for cat, rate in loss_rates.items():
            print(f"{cat}: {rate*100:.2f}%")
        
        return loss_rates
        
    except FileNotFoundError:
        print("警告：找不到 file_14.csv 文件，使用默认损耗率")
        # 备用方案：使用默认损耗率
        default_loss_rates = {
            '水生根茎类': 0.05,    # 5%损耗率
            '花叶类': 0.08,       # 8%损耗率
            '花菜类': 0.06,       # 6%损耗率
            '茄类': 0.07,         # 7%损耗率
            '辣椒类': 0.06,       # 6%损耗率
            '食用菌': 0.10        # 10%损耗率
        }
        
        print("使用默认损耗率：")
        for cat, rate in default_loss_rates.items():
            print(f"{cat}: {rate*100:.1f}%")
        
        return default_loss_rates

def analyze_loss_rates():
    """分析损耗率数据的详细统计信息"""
    try:
        # 读取损耗率数据
        loss_data = pd.read_csv('file_14.csv')
        
        print("\n损耗率数据详细分析：")
        print("=" * 60)
        
        # 按分类统计损耗率
        loss_stats = loss_data.groupby('分类名称')['损耗率(%)'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(2)
        
        print("各分类损耗率统计：")
        print(loss_stats.to_string())
        
        # 显示损耗率分布
        print(f"\n总体损耗率统计：")
        print(f"总商品数: {len(loss_data)}")
        print(f"平均损耗率: {loss_data['损耗率(%)'].mean():.2f}%")
        print(f"损耗率标准差: {loss_data['损耗率(%)'].std():.2f}%")
        print(f"最小损耗率: {loss_data['损耗率(%)'].min():.2f}%")
        print(f"最大损耗率: {loss_data['损耗率(%)'].max():.2f}%")
        
        # 显示损耗率较高的商品
        high_loss = loss_data[loss_data['损耗率(%)'] > 15].sort_values('损耗率(%)', ascending=False)
        if len(high_loss) > 0:
            print(f"\n损耗率超过15%的商品（共{len(high_loss)}个）:")
            for _, row in high_loss.head(10).iterrows():
                print(f"  {row['单品名称_x']} ({row['分类名称']}): {row['损耗率(%)']:.2f}%")
        
        return loss_stats
        
    except FileNotFoundError:
        print("无法分析损耗率数据：找不到 file_14.csv 文件")
        return None

# 注意：以下两个函数已被替换，现在直接使用需求预测结果.csv中的数据
# def forecast_wholesale_prices(historical_stats, forecast_dates):
#     """预测批发价格（已替换为使用需求预测结果.csv）"""
#     pass

# def forecast_base_demand(historical_stats, forecast_dates):
#     """预测基础需求（已替换为使用需求预测结果.csv）"""
#     pass

def optimize_pricing_and_replenishment(category, a, b, P_avg, Q_base, C_wholesale, L):
    """
    优化单个品类的定价和补货量（线性需求函数）
    
    参数:
    - category: 分类名称
    - a: 需求函数截距
    - b: 需求函数斜率（绝对值）
    - P_avg: 历史平均价格
    - Q_base: 基础需求预测
    - C_wholesale: 批发价格
    - L: 损耗率
    
    返回:
    - P_opt: 最优价格
    - Q_opt: 最优需求
    - S_opt: 最优补货量
    - profit: 预期利润
    """
    
    # 有效成本（考虑损耗）
    C_eff = C_wholesale / (1 - L)
    
    # 需求函数：Q(P) = Q_base - b * (P - P_avg)
    # 或者使用：Q(P) = a - b * P
    
    # 方法1：基于历史平均价格调整的需求函数
    def demand_function(P):
        Q = Q_base - b * (P - P_avg)
        return max(Q, 0.1)  # 确保需求至少为0.1，避免负值
    
    # 方法2：直接使用线性回归的需求函数
    def demand_function_direct(P):
        Q = a - b * P
        return max(Q, 0.1)  # 确保需求至少为0.1，避免负值
    
    # 选择更好的需求函数（基于R²）
    # 这里暂时使用方法1，因为它考虑了历史平均价格
    
    # 目标函数：最大化利润
    def objective(P):
        Q = demand_function(P)
        if Q <= 0:  # 需求不能为负
            return 1e6  # 返回很大的正数，表示这个价格不可行
        # 利润 = 需求 × (价格 - 有效成本)
        profit = Q * (P - C_eff)
        return -profit  # 最小化负利润 = 最大化利润
    
    # 约束条件
    bounds = [(C_wholesale * 1.1, C_wholesale * 10.0)]  # 价格范围：批发价的1.1-10倍
    
    # 初始猜测：批发价的1.5倍
    x0 = [C_wholesale * 1.5]
    
    # 优化
    try:
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            P_opt = result.x[0]
            Q_opt = demand_function(P_opt)
            S_opt = Q_opt / (1 - L)  # 考虑损耗的补货量
            profit = -result.fun  # 转换回正利润
            
            return P_opt, Q_opt, S_opt, profit
        else:
            print(f"优化失败: {category}")
            return None, None, None, 0
            
    except Exception as e:
        print(f"优化异常 {category}: {e}")
        return None, None, None, 0

def optimize_pricing_and_replenishment_loglog(category, intercept, price_elasticity, P_avg, Q_base, C_wholesale, L):
    """
    优化单个品类的定价和补货量（Log-Log需求函数）
    
    参数:
    - category: 分类名称
    - intercept: log-log回归截距
    - price_elasticity: 价格弹性（负值）
    - P_avg: 历史平均价格
    - Q_base: 基础需求预测
    - C_wholesale: 批发价格
    - L: 损耗率
    
    返回:
    - P_opt: 最优价格
    - Q_opt: 最优需求
    - S_opt: 最优补货量
    - profit: 预期利润
    """
    
    # 有效成本（考虑损耗）
    C_eff = C_wholesale / (1 - L)
    
    # Log-Log需求函数：log(Q) = intercept + price_elasticity * log(P)
    # 转换为：Q(P) = exp(intercept) * P^price_elasticity
    # 调整基线：Q(P) = Q_base * (P/P_avg)^price_elasticity
    
    def demand_function(P):
        if P <= 0:
            return 0.1
        # 基于历史平均价格调整的需求函数
        Q = Q_base * (P / P_avg) ** price_elasticity
        return max(Q, 0.1)  # 确保需求至少为0.1
    
    # 目标函数：最大化利润
    def objective(P):
        Q = demand_function(P)
        if Q <= 0:
            return 1e6
        # 利润 = 需求 × (价格 - 有效成本)
        profit = Q * (P - C_eff)
        return -profit  # 最小化负利润 = 最大化利润
    
    # 约束条件
    bounds = [(C_wholesale * 1.1, C_wholesale * 10.0)]
    
    # 初始猜测：批发价的1.5倍
    x0 = [C_wholesale * 1.5]
    
    # 优化
    try:
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            P_opt = result.x[0]
            Q_opt = demand_function(P_opt)
            S_opt = Q_opt / (1 - L)
            profit = -result.fun
            
            return P_opt, Q_opt, S_opt, profit
        else:
            print(f"Log-Log优化失败: {category}")
            return None, None, None, 0
            
    except Exception as e:
        print(f"Log-Log优化异常 {category}: {e}")
        return None, None, None, 0

def run_optimization():
    """运行完整的优化过程"""
    print("=" * 60)
    print("开始Q2步骤3：定价和补货最优化")
    print("=" * 60)
    
    # 1. 加载回归结果
    linear_results, loglog_results = load_regression_results()
    if linear_results is None or loglog_results is None:
        return
    
    # 2. 加载历史数据
    historical_stats, pq_data = load_historical_data()
    if historical_stats is None:
        return
    
    # 3. 设置损耗率
    print("\n分析损耗率数据...")
    loss_stats = analyze_loss_rates()
    loss_rates = estimate_loss_rates()
    
    # 4. 加载预测数据
    print("\n加载需求预测数据...")
    forecast_dict, forecast_data = load_forecast_data()
    if forecast_dict is None:
        return
    
    # 获取预测日期
    forecast_dates = sorted(forecast_data['日期'].unique())
    print(f"预测期间: {forecast_dates[0].strftime('%Y-%m-%d')} 到 {forecast_dates[-1].strftime('%Y-%m-%d')}")
    
    # 5. 使用已有的预测数据（不再需要重新预测）
    
    # 6. 运行优化
    optimization_results = []
    total_profit = 0
    
    print("\n开始优化每个分类的定价和补货策略...")
    print("-" * 80)
    
    for _, row in linear_results.iterrows():
        category = row['分类名称']
        a = row['intercept']
        b = abs(row['slope'])  # 取绝对值，因为回归结果中斜率通常为负
        
        # 获取log-log回归参数
        loglog_row = loglog_results[loglog_results['分类名称'] == category]
        if len(loglog_row) > 0:
            loglog_intercept = loglog_row.iloc[0]['intercept']
            price_elasticity = loglog_row.iloc[0]['price_elasticity']
        else:
            print(f"警告：找不到 {category} 的log-log回归结果")
            continue
        
        # 获取历史统计
        hist_row = historical_stats[historical_stats['分类名称'] == category].iloc[0]
        P_avg = hist_row['P_avg']
        Q_avg = hist_row['Q_avg']
        
        # 获取损耗率
        L = loss_rates[category]
        
        print(f"\n优化分类: {category}")
        print(f"线性模型参数: a={a:.2f}, b={b:.4f}")
        print(f"Log-Log模型参数: intercept={loglog_intercept:.4f}, 价格弹性={price_elasticity:.4f}")
        print(f"历史平均价格: {P_avg:.2f}, 历史平均销量: {Q_avg:.2f}")
        print(f"损耗率: {L*100:.1f}%")
        
        category_results = []
        category_profit_linear = 0
        category_profit_loglog = 0
        
        for j, date in enumerate(forecast_dates):
            # 从预测数据中获取基础需求和批发价格
            forecast_info = get_forecast_for_date(forecast_dict, category, date)
            if forecast_info is None:
                continue
                
            Q_base = forecast_info['demand']
            C_wholesale = forecast_info['wholesale_price']
            
            # 使用预测数据中的损耗率（如果可用），否则使用file_14.csv中的损耗率
            L_forecast = forecast_info['loss_rate']
            L_used = L_forecast if L_forecast > 0 else L
            
            # 运行线性模型优化
            P_opt_linear, Q_opt_linear, S_opt_linear, profit_linear = optimize_pricing_and_replenishment(
                category, a, b, P_avg, Q_base, C_wholesale, L_used
            )
            
            # 运行Log-Log模型优化
            P_opt_loglog, Q_opt_loglog, S_opt_loglog, profit_loglog = optimize_pricing_and_replenishment_loglog(
                category, loglog_intercept, price_elasticity, P_avg, Q_base, C_wholesale, L_used
            )
            
            if P_opt_linear is not None and P_opt_loglog is not None:
                # 线性模型结果
                result_linear = {
                    '分类名称': category,
                    '日期': date.strftime('%Y-%m-%d'),
                    '模型类型': '线性模型',
                    '基础需求预测': round(Q_base, 2),
                    '批发价格预测': round(C_wholesale, 2),
                    '最优定价': round(P_opt_linear, 2),
                    '最优需求预测': round(Q_opt_linear, 2),
                    '最优补货量': round(S_opt_linear, 2),
                    '预期利润': round(profit_linear, 2),
                    '使用损耗率': f"{L_used*100:.1f}%",
                    'file_14损耗率': f"{L*100:.1f}%",
                    '预测损耗率': f"{L_forecast*100:.1f}%"
                }
                
                # Log-Log模型结果
                result_loglog = {
                    '分类名称': category,
                    '日期': date.strftime('%Y-%m-%d'),
                    '模型类型': 'Log-Log模型',
                    '基础需求预测': round(Q_base, 2),
                    '批发价格预测': round(C_wholesale, 2),
                    '最优定价': round(P_opt_loglog, 2),
                    '最优需求预测': round(Q_opt_loglog, 2),
                    '最优补货量': round(S_opt_loglog, 2),
                    '预期利润': round(profit_loglog, 2),
                    '使用损耗率': f"{L_used*100:.1f}%",
                    'file_14损耗率': f"{L*100:.1f}%",
                    '预测损耗率': f"{L_forecast*100:.1f}%"
                }
                
                category_results.extend([result_linear, result_loglog])
                category_profit_linear += profit_linear
                category_profit_loglog += profit_loglog
                
                print(f"  {date.strftime('%m-%d')}:")
                print(f"    线性模型: P={P_opt_linear:.2f}, Q={Q_opt_linear:.1f}, S={S_opt_linear:.1f}, 利润={profit_linear:.1f}")
                print(f"    Log-Log模型: P={P_opt_loglog:.2f}, Q={Q_opt_loglog:.1f}, S={S_opt_loglog:.1f}, 利润={profit_loglog:.1f}")
            else:
                print(f"  {date.strftime('%m-%d')}: 优化失败")
        
        # 添加到总结果
        optimization_results.extend(category_results)
        total_profit += max(category_profit_linear, category_profit_loglog)  # 选择更好的模型
        
        print(f"分类总利润 - 线性模型: {category_profit_linear:.1f}, Log-Log模型: {category_profit_loglog:.1f}")
        print("-" * 40)
    
    # 7. 保存结果
    if optimization_results:
        results_df = pd.DataFrame(optimization_results)
        results_df.to_csv('Q2_optimization_results.csv', index=False, encoding='utf-8-sig')
        
        # 模型对比分析
        print("\n" + "=" * 60)
        print("模型对比分析")
        print("=" * 60)
        
        # 按模型类型统计利润
        model_comparison = results_df.groupby(['分类名称', '模型类型'])['预期利润'].sum().unstack(fill_value=0)
        print("\n各分类两种模型的利润对比:")
        print(model_comparison.to_string())
        
        # 计算总体利润
        total_profit_linear = results_df[results_df['模型类型'] == '线性模型']['预期利润'].sum()
        total_profit_loglog = results_df[results_df['模型类型'] == 'Log-Log模型']['预期利润'].sum()
        
        print(f"\n总体利润对比:")
        print(f"线性模型总利润: {total_profit_linear:.1f} 元")
        print(f"Log-Log模型总利润: {total_profit_loglog:.1f} 元")
        print(f"差异: {abs(total_profit_linear - total_profit_loglog):.1f} 元")
        
        # 选择更好的模型
        better_model = "线性模型" if total_profit_linear > total_profit_loglog else "Log-Log模型"
        print(f"推荐模型: {better_model}")
        
        print("\n" + "=" * 60)
        print("优化完成！")
        print("=" * 60)
        print(f"一周总预期利润: {max(total_profit_linear, total_profit_loglog):.1f} 元")
        print(f"结果已保存至: Q2_optimization_results.csv")
        
        # 8. 创建结果汇总
        create_optimization_summary(results_df, max(total_profit_linear, total_profit_loglog))
        
        return results_df
    else:
        print("优化失败，没有生成结果")
        return None

def create_optimization_summary(results_df, total_profit):
    """创建优化结果汇总图表"""
    print("\n创建优化结果汇总图表...")
    
    # 创建更大的图表布局
    plt.figure(figsize=(16, 12))
    
    # 1. 两种模型的利润对比
    plt.subplot(3, 3, 1)
    model_profits = results_df.groupby('模型类型')['预期利润'].sum()
    colors = ['blue', 'orange']
    bars = plt.bar(model_profits.index, model_profits.values, color=colors)
    plt.xlabel('模型类型')
    plt.ylabel('一周总利润 (元)')
    plt.title('两种模型利润对比')
    for i, v in enumerate(model_profits.values):
        plt.text(i, v, f'{v:.0f}', ha='center', va='bottom')
    
    # 2. 各分类两种模型利润对比
    plt.subplot(3, 3, 2)
    category_model_profits = results_df.groupby(['分类名称', '模型类型'])['预期利润'].sum().unstack(fill_value=0)
    x = np.arange(len(category_model_profits))
    width = 0.35
    
    plt.bar(x - width/2, category_model_profits['线性模型'], width, label='线性模型', color='blue', alpha=0.7)
    plt.bar(x + width/2, category_model_profits['Log-Log模型'], width, label='Log-Log模型', color='orange', alpha=0.7)
    
    plt.xlabel('分类')
    plt.ylabel('一周总利润 (元)')
    plt.title('各分类两种模型利润对比')
    plt.xticks(x, category_model_profits.index, rotation=45)
    plt.legend()
    
    # 3. 各分类最优定价对比
    plt.subplot(3, 3, 3)
    price_comparison = results_df.groupby(['分类名称', '模型类型'])['最优定价'].mean().unstack(fill_value=0)
    x = np.arange(len(price_comparison))
    
    plt.bar(x - width/2, price_comparison['线性模型'], width, label='线性模型', color='blue', alpha=0.7)
    plt.bar(x + width/2, price_comparison['Log-Log模型'], width, label='Log-Log模型', color='orange', alpha=0.7)
    
    plt.xlabel('分类')
    plt.ylabel('平均最优定价 (元/千克)')
    plt.title('各分类两种模型定价对比')
    plt.xticks(x, price_comparison.index, rotation=45)
    plt.legend()
    
    # 4. 各分类补货量对比
    plt.subplot(3, 3, 4)
    replenishment_comparison = results_df.groupby(['分类名称', '模型类型'])['最优补货量'].sum().unstack(fill_value=0)
    x = np.arange(len(replenishment_comparison))
    
    plt.bar(x - width/2, replenishment_comparison['线性模型'], width, label='线性模型', color='blue', alpha=0.7)
    plt.bar(x + width/2, replenishment_comparison['Log-Log模型'], width, label='Log-Log模型', color='orange', alpha=0.7)
    
    plt.xlabel('分类')
    plt.ylabel('一周总补货量 (千克)')
    plt.title('各分类两种模型补货量对比')
    plt.xticks(x, replenishment_comparison.index, rotation=45)
    plt.legend()
    
    # 5. 利润时间趋势对比
    plt.subplot(3, 3, 5)
    daily_profits_comparison = results_df.groupby(['日期', '模型类型'])['预期利润'].sum().unstack(fill_value=0)
    dates = [d[5:] for d in daily_profits_comparison.index]
    
    plt.plot(range(len(dates)), daily_profits_comparison['线性模型'], marker='o', linewidth=2, markersize=6, label='线性模型', color='blue')
    plt.plot(range(len(dates)), daily_profits_comparison['Log-Log模型'], marker='s', linewidth=2, markersize=6, label='Log-Log模型', color='orange')
    
    plt.xlabel('日期')
    plt.ylabel('日利润 (元)')
    plt.title('一周利润时间趋势对比')
    plt.xticks(range(len(dates)), dates, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. 模型性能雷达图
    plt.subplot(3, 3, 6)
    # 计算各分类的模型性能指标
    performance_metrics = {}
    for category in results_df['分类名称'].unique():
        cat_data = results_df[results_df['分类名称'] == category]
        linear_profit = cat_data[cat_data['模型类型'] == '线性模型']['预期利润'].sum()
        loglog_profit = cat_data[cat_data['模型类型'] == 'Log-Log模型']['预期利润'].sum()
        performance_metrics[category] = {
            '线性模型': linear_profit,
            'Log-Log模型': loglog_profit
        }
    
    # 选择前4个分类进行雷达图展示
    top_categories = sorted(performance_metrics.items(), key=lambda x: max(x[1].values()), reverse=True)[:4]
    
    if len(top_categories) >= 2:
        categories = [cat[0] for cat in top_categories]
        linear_values = [cat[1]['线性模型'] for cat in top_categories]
        loglog_values = [cat[1]['Log-Log模型'] for cat in top_categories]
        
        # 标准化到0-1范围
        max_val = max(max(linear_values), max(loglog_values))
        linear_norm = [v/max_val for v in linear_values]
        loglog_norm = [v/max_val for v in loglog_values]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        linear_norm += linear_norm[:1]
        loglog_norm += loglog_norm[:1]
        
        ax = plt.subplot(3, 3, 6, projection='polar')
        ax.plot(angles, linear_norm, 'o-', linewidth=2, label='线性模型', color='blue')
        ax.fill(angles, linear_norm, alpha=0.25, color='blue')
        ax.plot(angles, loglog_norm, 'o-', linewidth=2, label='Log-Log模型', color='orange')
        ax.fill(angles, loglog_norm, alpha=0.25, color='orange')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        plt.title('模型性能雷达图对比')
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 7. 价格弹性分析
    plt.subplot(3, 3, 7)
    # 这里可以添加价格弹性相关的分析
    plt.text(0.5, 0.5, '价格弹性分析\n(基于Log-Log模型)', ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.axis('off')
    
    # 8. 损耗率影响分析
    plt.subplot(3, 3, 8)
    loss_impact = results_df.groupby('分类名称')['使用损耗率'].first().str.rstrip('%').astype(float)
    plt.bar(range(len(loss_impact)), loss_impact.values, color='red', alpha=0.7)
    plt.xlabel('分类')
    plt.ylabel('损耗率 (%)')
    plt.title('各分类损耗率')
    plt.xticks(range(len(loss_impact)), loss_impact.index, rotation=45)
    
    # 9. 模型选择建议
    plt.subplot(3, 3, 9)
    better_model_by_category = []
    for category in results_df['分类名称'].unique():
        cat_data = results_df[results_df['分类名称'] == category]
        linear_profit = cat_data[cat_data['模型类型'] == '线性模型']['预期利润'].sum()
        loglog_profit = cat_data[cat_data['模型类型'] == 'Log-Log模型']['预期利润'].sum()
        better_model = '线性模型' if linear_profit > loglog_profit else 'Log-Log模型'
        better_model_by_category.append(better_model)
    
    model_counts = pd.Series(better_model_by_category).value_counts()
    plt.pie(model_counts.values, labels=model_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('各分类模型选择建议')
    
    plt.tight_layout()
    plt.savefig('Q2_optimization_summary.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("优化结果汇总图已保存为: Q2_optimization_summary.png")

def main():
    """主函数"""
    print("Q2步骤3：定价和补货最优化模型")
    print("基于线性回归需求函数：Q = a - b * P")
    print("优化目标：最大化一周总收益")
    
    # 运行优化
    results = run_optimization()
    
    if results is not None:
        print("\n优化结果预览（前10行）:")
        print(results.head(10).to_string(index=False))
        
        print(f"\n完整结果已保存至: Q2_optimization_results.csv")
        print("汇总图表已保存至: Q2_optimization_summary.png")

if __name__ == '__main__':
    main()