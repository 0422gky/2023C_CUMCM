# 分析file_23_r_with_category_consolidated.csv中p和每日品类销量的关系
# 先做销售单价列和销量之间的关联,按分类名称分组

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 确保字体设置生效
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_and_analyze_data():
    """加载数据并分析销售单价与销量的关系"""
    
    # 读取CSV文件
    print("正在读取数据...")
    df = pd.read_csv('file_23_r_with_category_consolidated.csv')
    
    # 转换销售日期为datetime类型
    df['销售日期'] = pd.to_datetime(df['销售日期'])
    
    print(f"数据加载完成，共 {len(df)} 条记录")
    print(f"日期范围：{df['销售日期'].min()} 到 {df['销售日期'].max()}")
    print(f"分类名称：{df['分类名称'].unique()}")
    
    return df

def analyze_price_sales_correlation(df):
    """分析销售单价与销量的相关性"""
    
    print("\n=== 销售单价与销量相关性分析 ===")
    
    # 按分类名称分组计算相关性
    correlations = {}
    for category in df['分类名称'].unique():
        cat_data = df[df['分类名称'] == category]
        
        # 计算皮尔逊相关系数
        corr, p_value = stats.pearsonr(cat_data['销售单价(元/千克)'], cat_data['销量(千克)'])
        
        correlations[category] = {
            'correlation': corr,
            'p_value': p_value,
            'count': len(cat_data)
        }
        
        print(f"\n{category}:")
        print(f"  相关系数: {corr:.4f}")
        print(f"  P值: {p_value:.4f}")
        print(f"  样本数量: {len(cat_data)}")
    
    return correlations

def analyze_by_date_category(df):
    """按销售日期和分类名称分组分析"""
    
    print("\n=== 按日期和分类名称分组分析 ===")
    
    # 按日期和分类名称分组，计算每日每类的平均价格和总销量
    daily_category_stats = df.groupby(['销售日期', '分类名称']).agg({
        '销售单价(元/千克)': ['mean', 'std', 'min', 'max'],
        '销量(千克)': ['sum', 'mean', 'count'],
        '批发价格(元/千克)': 'mean'
    }).round(4)
    
    # 重命名列
    daily_category_stats.columns = [
        '平均销售单价', '销售单价标准差', '最低销售单价', '最高销售单价',
        '总销量', '平均销量', '商品数量',
        '平均批发价格'
    ]
    
    print("每日每类统计信息（前10行）:")
    print(daily_category_stats.head(10))
    
    return daily_category_stats

def create_visualizations(df, daily_category_stats):
    """创建可视化图表"""
    
    print("\n=== 创建可视化图表 ===")
    
    # 设置图表样式和字体
    plt.style.use('seaborn-v0_8')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 各分类的销售单价分布
    plt.figure(figsize=(12, 8))
    df.boxplot(column='销售单价(元/千克)', by='分类名称')
    plt.title('各分类销售单价分布', fontsize=16, fontweight='bold')
    plt.xlabel('分类名称', fontsize=12)
    plt.ylabel('销售单价(元/千克)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('各分类销售单价分布.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 各分类的销量分布
    plt.figure(figsize=(12, 8))
    df.boxplot(column='销量(千克)', by='分类名称')
    plt.title('各分类销量分布', fontsize=16, fontweight='bold')
    plt.xlabel('分类名称', fontsize=12)
    plt.ylabel('销量(千克)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('各分类销量分布.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 销售单价与销量的散点图（按分类）
    plt.figure(figsize=(12, 8))
    for category in df['分类名称'].unique():
        cat_data = df[df['分类名称'] == category]
        plt.scatter(cat_data['销售单价(元/千克)'], cat_data['销量(千克)'], 
                   alpha=0.6, label=category, s=20)
    
    plt.xlabel('销售单价(元/千克)', fontsize=12)
    plt.ylabel('销量(千克)', fontsize=12)
    plt.title('销售单价与销量散点图', fontsize=16, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('销售单价与销量散点图.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 每日平均价格趋势
    plt.figure(figsize=(12, 8))
    daily_price = df.groupby('销售日期')['销售单价(元/千克)'].mean()
    daily_price.plot(marker='o', markersize=3, alpha=0.7, linewidth=1.5)
    plt.title('每日平均销售单价趋势', fontsize=16, fontweight='bold')
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('平均销售单价(元/千克)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('每日平均销售单价趋势.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建相关性热力图
    create_correlation_heatmap(df)

def create_correlation_heatmap(df):
    """创建相关性热力图"""
    
    # 选择数值列进行相关性分析
    numeric_cols = ['销量(千克)', '销售单价(元/千克)', '批发价格(元/千克)', 'r_i,t', 'lambda_i,t']
    
    # 计算相关性矩阵
    corr_matrix = df[numeric_cols].corr()
    
    # 设置字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 先画热图但不自动注释
    sns.heatmap(corr_matrix, annot=False, cmap='vlag', center=0, ax=ax, 
                linewidths=0.5, linecolor='white', cbar=True)
    
    # 计算阈值用于选择注释颜色（参考Q1.py的方法）
    vmin = np.nanmin(corr_matrix.values)
    vmax = np.nanmax(corr_matrix.values)
    thresh = (vmax + vmin) / 2.0
    
    # 逐单元写入注释，按阈值选择白/黑
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            val = corr_matrix.iloc[i, j]
            if pd.isna(val):
                continue
            # 对称阈值：数值越大用白字，越小用黑字
            color = 'white' if val > thresh else 'black'
            ax.text(j + 0.5, i + 0.5, f"{val:.2f}", ha='center', va='center', 
                   color=color, fontsize=9)
    
    ax.set_title('数值变量相关性热力图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('相关性热力图.png', dpi=150, bbox_inches='tight')
    plt.close()

def log_log_price_elasticity(df, category_col='分类名称',
                             price_col='销售单价(元/千克)',
                             qty_col='销量(千克)'):
    """使用log-log回归方法计算价格弹性
    
    价格弹性计算公式：logQ = α + β logP
    其中：
    - β 是价格弹性系数
    - logQ 是销量的对数值
    - logP 是价格的对数值
    
    价格弹性解释：
    - |β| > 1: 富有弹性（价格变化对销量影响大）
    - |β| = 1: 单位弹性
    - |β| < 1: 缺乏弹性（价格变化对销量影响小）
    - β < 0: 正常商品（价格上升，销量下降）
    """
    
    print("\n=== 价格弹性分析（Log-Log回归方法）===")
    
    results = {}
    
    for category in df[category_col].unique():
        cat_data = df[df[category_col] == category].copy()
        
        # 去掉价格或销量为0或负数的点（log无法取值）
        cat_data = cat_data[(cat_data[price_col] > 0) & (cat_data[qty_col] > 0)]
        
        if len(cat_data) > 10:  # 确保数据量足够
            # 对数变换
            cat_data['logP'] = np.log(cat_data[price_col])
            cat_data['logQ'] = np.log(cat_data[qty_col])
            
            # 回归：logQ = α + β logP
            X = sm.add_constant(cat_data['logP'])
            y = cat_data['logQ']
            model = sm.OLS(y, X).fit()
            
            beta = model.params['logP']  # 价格弹性
            r_squared = model.rsquared
            p_value = model.pvalues['logP']
            
            results[category] = {
                '价格弹性': beta,
                'R方': r_squared,
                'P值': p_value,
                '样本数量': len(cat_data),
                '回归摘要': model.summary()
            }
            
            print(f"\n{category}:")
            print(f"  价格弹性: {beta:.4f}")
            print(f"  R方: {r_squared:.4f}")
            print(f"  P值: {p_value:.4f}")
            print(f"  样本数量: {len(cat_data)}")
            print(f"  回归摘要: {model.summary()}")
        else:
            print(f"\n{category}: 数据量不足（{len(cat_data)} < 10），跳过分析")
    
    return results

def main():
    """主函数"""
    
    try:
        # 加载数据
        df = load_and_analyze_data()
        
        # 分析相关性
        correlations = analyze_price_sales_correlation(df)
        
        # 按日期和分类名称分组分析
        daily_category_stats = analyze_by_date_category(df)
        
        # 价格弹性分析
        elasticity_results = log_log_price_elasticity(df)
        
        # 创建可视化
        create_visualizations(df, daily_category_stats)
        
        # 保存分析结果
        daily_category_stats.to_csv('每日每类统计分析.csv', encoding='utf-8-sig')
        
        # 保存相关性分析结果
        correlations_df = pd.DataFrame.from_dict(correlations, orient='index')
        correlations_df.to_csv('销售单价与销量相关性分析.csv', encoding='utf-8-sig')
        
        # 保存价格弹性分析结果
        if elasticity_results:
            # 提取主要指标，排除回归摘要（因为包含复杂对象）
            elasticity_summary = {}
            for category, result in elasticity_results.items():
                elasticity_summary[category] = {
                    '价格弹性': result['价格弹性'],
                    'R方': result['R方'],
                    'P值': result['P值'],
                    '样本数量': result['样本数量']
                }
            
            elasticity_df = pd.DataFrame.from_dict(elasticity_summary, orient='index')
            elasticity_df.to_csv('价格弹性分析结果.csv', encoding='utf-8-sig')
        
        print("\n=== 分析完成 ===")
        print("结果已保存到:")
        print("- 每日每类统计分析.csv")
        print("- 销售单价与销量相关性分析.csv")
        print("- 价格弹性分析结果.csv")
        print("- 各分类销售单价分布.png")
        print("- 各分类销量分布.png")
        print("- 销售单价与销量散点图.png")
        print("- 每日平均销售单价趋势.png")
        print("- 相关性热力图.png")
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
