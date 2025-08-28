# -*- coding: utf-8 -*-
"""
价格与销量关系回归分析：Log-Log回归 + 线性回归
P_i(t) = Σ(p_j(t) * q_j(t)) / Σ(q_j(t)) j∈i
Q_i(t) = Σ(q_j(t)) j∈i
"""

import os
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def prepare_pq(input_csv='file_23_r_with_category_consolidated.csv', out_csv='file_23_pq.csv'):
    """准备P和Q数据，计算加权平均价格"""
    expected_cols = ['销售日期','单品编码','分类名称','销量(千克)','销售单价(元/千克)','批发价格(元/千克)','r_i,t','每日品类销量','lambda_i,t']
    df = pd.read_csv(input_csv, dtype=str)
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        print('警告：输入文件缺少下列预期列（将继续但可能失败）:', missing)

    # 列名确定
    price_col = '销售单价(元/千克)'
    if '每日品类销量' in df.columns:
        qty_col = '每日品类销量'
    else:
        qty_col = '销量(千克)'

    # 转类型并处理
    df['销售日期'] = pd.to_datetime(df['销售日期'])
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce')

    # 按分类名称+销售日期聚合：加权价格P和总销量Q
    grp = df.groupby(['分类名称', '销售日期']).apply(
        lambda x: pd.Series({
            'P': (x[price_col] * x[qty_col]).sum() / x[qty_col].sum() if x[qty_col].sum() > 0 else np.nan,
            'Q': x[qty_col].mean()
        })
    ).reset_index()

    grp.to_csv(out_csv, index=False, encoding='utf-8-sig')
    return grp

def run_linear_regressions(pq_df):
    """执行线性回归分析：Q ~ P"""
    results = {}
    categories = pq_df['分类名称'].unique()
    os.makedirs('regression_plots', exist_ok=True)
    rows = []
    
    for cat in categories:
        sub = pq_df[pq_df['分类名称'] == cat].dropna(subset=['P', 'Q']).copy()
        
        if len(sub) < 5:
            print(f'分类 {cat} 数据点过少，跳过')
            continue
            
        # 线性回归分析：Q ~ P
        try:
            model = smf.ols('Q ~ P', data=sub).fit()
            results[cat] = model
            
            print(f'==== 线性回归结果: {cat} ====')
            print(model.summary())
            
            # 保存摘要要素
            coef = model.params.to_dict()
            pvalues = model.pvalues.to_dict()
            r2 = model.rsquared
            
            # 计算线性回归系数
            slope = coef.get('P', np.nan)
            intercept = coef.get('Intercept', np.nan)
            
            rows.append({
                '分类名称': cat,
                'n': len(sub),
                'intercept': intercept,
                'slope': slope,
                'p_intercept': pvalues.get('Intercept', np.nan),
                'p_slope': pvalues.get('P', np.nan),
                'r2': r2,
                'regression_type': '线性回归'
            })

            # 绘制线性回归图
            create_linear_regression_plots(sub, model, coef, cat)
            
        except Exception as e:
            print('线性回归失败', cat, e)
            import traceback
            traceback.print_exc()
    
    # 保存线性回归结果表
    if rows:
        outdf = pd.DataFrame(rows)
        outdf.to_csv('linear_regression_results.csv', index=False, encoding='utf-8-sig')
        print('线性回归结果已保存为 linear_regression_results.csv')
        
        # 打印线性回归汇总
        print('\n==== 线性回归汇总 ====')
        for row in rows:
            cat = row['分类名称']
            slope = row['slope']
            r2 = row['r2']
            print(f'{cat}: 斜率 = {slope:.6f}, R² = {r2:.4f}')
    


def run_log_log_regressions(pq_df):
    """执行log-log回归分析：log(Q) ~ log(P)"""
    results = {}
    categories = pq_df['分类名称'].unique()
    os.makedirs('regression_plots', exist_ok=True)
    rows = []
    
    for cat in categories:
        sub = pq_df[pq_df['分类名称'] == cat].dropna(subset=['P', 'Q']).copy()
        
        # 过滤掉价格和销量为0或负数的数据点（因为要取对数）
        sub = sub[(sub['P'] > 0) & (sub['Q'] > 0)]
        
        if len(sub) < 5:
            print(f'分类 {cat} 数据点过少，跳过')
            continue
            
        # 计算对数变换
        sub['log_P'] = np.log(sub['P'])
        sub['log_Q'] = np.log(sub['Q'])
        
        # log-log回归分析：log(Q) ~ log(P)
        try:
            model = smf.ols('log_Q ~ log_P', data=sub).fit()
            results[cat] = model
            
            print(f'==== Log-Log回归结果: {cat} ====')
            print(model.summary())
            
            # 保存摘要要素
            coef = model.params.to_dict()
            pvalues = model.pvalues.to_dict()
            r2 = model.rsquared
            
            # 计算价格弹性（log-log回归的斜率就是价格弹性）
            price_elasticity = coef.get('log_P', np.nan)
            
            rows.append({
                '分类名称': cat,
                'n': len(sub),
                'intercept': coef.get('Intercept', np.nan),
                'log_P_coef': coef.get('log_P', np.nan),
                'price_elasticity': price_elasticity,
                'p_intercept': pvalues.get('Intercept', np.nan),
                'p_log_P': pvalues.get('log_P', np.nan),
                'r2': r2,
                'elasticity_interpretation': interpret_elasticity(price_elasticity),
                'regression_type': 'Log-Log回归'
            })

            # 绘制log-log散点图和回归线
            create_loglog_regression_plots(sub, model, coef, price_elasticity, cat)
            
        except Exception as e:
            print('Log-Log回归失败', cat, e)
            import traceback
            traceback.print_exc()
    
    # 保存log-log回归结果表
    if rows:
        outdf = pd.DataFrame(rows)
        outdf.to_csv('loglog_regression_results.csv', index=False, encoding='utf-8-sig')
        print('Log-Log回归结果已保存为 loglog_regression_results.csv')
        
        # 打印价格弹性汇总
        print('\n==== 价格弹性汇总 ====')
        for row in rows:
            cat = row['分类名称']
            elasticity = row['price_elasticity']
            interpretation = row['elasticity_interpretation']
            print(f'{cat}: 价格弹性 = {elasticity:.3f} ({interpretation})')
    
    return results

def create_linear_regression_plots(sub, model, coef, cat):
    """创建线性回归分析图表"""
    plt.figure(figsize=(8, 6))
    plt.scatter(sub['P'], sub['Q'], alpha=0.6, s=50)
    
    # 生成回归线
    x_min, x_max = sub['P'].min(), sub['P'].max()
    x_vals = np.linspace(x_min, x_max, 100)
    y_pred = model.predict(pd.DataFrame({'P': x_vals}))
    plt.plot(x_vals, y_pred, color='blue', linewidth=2, 
             label=f'回归线: Q = {coef["Intercept"]:.3f} + {coef["P"]:.6f}×P')
    
    plt.xlabel('P - 加权平均价格 (元/千克)')
    plt.ylabel('Q - 每日品类销量 (千克)')
    plt.title(f'分类 {cat}：线性回归分析\nQ = {coef["Intercept"]:.3f} + {coef["P"]:.6f}×P')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    safe_name = str(cat).replace('/', '_').replace('\\', '_')
    plt.savefig(f'regression_plots/linear_regression_{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_loglog_regression_plots(sub, model, coef, price_elasticity, cat):
    """创建log-log回归分析图表"""
    # log-log散点图 + 回归直线
    plt.figure(figsize=(8, 6))
    plt.scatter(sub['log_P'], sub['log_Q'], alpha=0.6, s=50)
    
    # 生成回归线
    x_min, x_max = sub['log_P'].min(), sub['log_P'].max()
    x_vals = np.linspace(x_min, x_max, 100)
    y_pred = model.predict(pd.DataFrame({'log_P': x_vals}))
    plt.plot(x_vals, y_pred, color='red', linewidth=2, 
             label=f'回归线: log(Q) = {coef["Intercept"]:.3f} + {coef["log_P"]:.3f}×log(P)')
    
    plt.xlabel('log(P) - 价格对数')
    plt.ylabel('log(Q) - 销量对数')
    plt.title(f'分类 {cat}：Log-Log回归分析\n价格弹性 = {price_elasticity:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    safe_name = str(cat).replace('/', '_').replace('\\', '_')
    plt.savefig(f'regression_plots/loglog_regression_{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 原始数据拟合曲线
    plt.figure(figsize=(8, 6))
    plt.scatter(sub['P'], sub['Q'], alpha=0.6, s=50)
    
    # 在原始数据上绘制拟合曲线
    P_vals = np.linspace(sub['P'].min(), sub['P'].max(), 100)
    log_P_vals = np.log(P_vals)
    log_Q_pred = model.predict(pd.DataFrame({'log_P': log_P_vals}))
    Q_pred = np.exp(log_Q_pred)
    plt.plot(P_vals, Q_pred, color='red', linewidth=2, 
             label=f'拟合曲线: Q = exp({coef["Intercept"]:.3f}) × P^{coef["log_P"]:.3f}')
    
    plt.xlabel('P - 加权平均价格 (元/千克)')
    plt.ylabel('Q - 每日品类销量 (千克)')
    plt.title(f'分类 {cat}：原始数据拟合曲线\n价格弹性 = {price_elasticity:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f'regression_plots/original_fit_{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

def interpret_elasticity(elasticity):
    """解释价格弹性的含义"""
    if np.isnan(elasticity):
        return "无法计算"
    
    elasticity = float(elasticity)
    
    if elasticity < -1:
        return "富有弹性 (价格上升1%，销量下降超过1%)"
    elif elasticity == -1:
        return "单位弹性 (价格上升1%，销量下降1%)"
    elif elasticity > -1 and elasticity < 0:
        return "缺乏弹性 (价格上升1%，销量下降少于1%)"
    elif elasticity == 0:
        return "完全无弹性 (价格变化不影响销量)"
    elif elasticity > 0:
        return "正弹性 (价格上升，销量也上升)"
    else:
        return "负弹性 (价格上升，销量下降)"

def create_comparison_plots():
    """创建两种回归方法的对比图表"""
    try:
        # 读取两种回归结果
        linear_df = pd.read_csv('linear_regression_results.csv')
        loglog_df = pd.read_csv('loglog_regression_results.csv')
        
        # 合并数据用于对比
        comparison_data = []
        for _, row in linear_df.iterrows():
            cat = row['分类名称']
            loglog_row = loglog_df[loglog_df['分类名称'] == cat]
            if not loglog_row.empty:
                comparison_data.append({
                    '分类名称': cat,
                    '线性回归_R2': row['r2'],
                    'LogLog回归_R2': loglog_row.iloc[0]['r2'],
                    '线性回归_斜率': row['slope'],
                    'LogLog回归_价格弹性': loglog_row.iloc[0]['price_elasticity']
                })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            
            # 创建R²对比图
            plt.figure(figsize=(12, 8))
            
            x = np.arange(len(comp_df))
            width = 0.35
            
            plt.bar(x - width/2, comp_df['线性回归_R2'], width, label='线性回归 R²', alpha=0.8)
            plt.bar(x + width/2, comp_df['LogLog回归_R2'], width, label='Log-Log回归 R²', alpha=0.8)
            
            plt.xlabel('分类')
            plt.ylabel('R² 决定系数')
            plt.title('线性回归 vs Log-Log回归 拟合效果对比')
            plt.xticks(x, comp_df['分类名称'], rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig('regression_plots/回归方法对比_R2.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # 保存对比结果
            comp_df.to_csv('regression_comparison_results.csv', index=False, encoding='utf-8-sig')
            print('回归方法对比结果已保存为 regression_comparison_results.csv')
            print('对比图表已保存为 regression_plots/回归方法对比_R2.png')
            
            # 打印对比结果
            print('\n==== 回归方法对比 ====')
            for _, row in comp_df.iterrows():
                cat = row['分类名称']
                linear_r2 = row['线性回归_R2']
                loglog_r2 = row['LogLog回归_R2']
                better_method = '线性回归' if linear_r2 > loglog_r2 else 'Log-Log回归'
                print(f'{cat}: 线性回归R²={linear_r2:.4f}, Log-Log回归R²={loglog_r2:.4f} (推荐: {better_method})')
                
    except Exception as e:
        print('创建对比图表失败:', e)

def create_elasticity_summary_plot(results_df):
    """创建价格弹性汇总图表"""
    if len(results_df) == 0:
        return
        
    plt.figure(figsize=(12, 8))
    
    # 按价格弹性排序
    results_sorted = results_df.sort_values('price_elasticity')
    
    # 创建条形图
    bars = plt.barh(range(len(results_sorted)), results_sorted['price_elasticity'], 
                    color=['red' if x < 0 else 'blue' for x in results_sorted['price_elasticity']])
    
    # 添加数值标签
    for i, (idx, row) in enumerate(results_sorted.iterrows()):
        plt.text(row['price_elasticity'], i, f'{row["price_elasticity"]:.3f}', 
                va='center', ha='left' if row['price_elasticity'] < 0 else 'right')
    
    plt.yticks(range(len(results_sorted)), results_sorted['分类名称'])
    plt.xlabel('价格弹性系数')
    plt.title('各分类价格弹性对比')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    plt.axvline(x=-1, color='red', linestyle='--', alpha=0.7, label='单位弹性线')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('regression_plots/价格弹性汇总对比.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("开始价格与销量关系回归分析...")
    print("=" * 60)
    
    # 准备数据
    pq = prepare_pq()
    
    # 执行线性回归分析
    print("\n1. 执行线性回归分析 (Q ~ P)...")
    linear_res = run_linear_regressions(pq)
    
    # 执行log-log回归分析
    print("\n2. 执行Log-Log回归分析 (log(Q) ~ log(P))...")
    loglog_res = run_log_log_regressions(pq)
    
    # 创建对比分析
    print("\n3. 创建回归方法对比分析...")
    create_comparison_plots()
    
    # 创建价格弹性汇总图
    try:
        results_df = pd.read_csv('loglog_regression_results.csv')
        create_elasticity_summary_plot(results_df)
        print("价格弹性汇总图已保存为 'regression_plots/价格弹性汇总对比.png'")
    except:
        print("无法创建价格弹性汇总图")
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    print("生成的文件:")
    print("- linear_regression_results.csv: 线性回归结果")
    print("- loglog_regression_results.csv: Log-Log回归结果")
    print("- regression_comparison_results.csv: 两种方法对比结果")
    print("- regression_plots/: 各种图表文件")

if __name__ == '__main__':
    main()
