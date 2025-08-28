# 研究file_2
# 第一问：研究各蔬菜品类销售总量和成本加成定价之间的关系
# 要对齐单品和品类之间的颗粒度，这里打算用一个权重系数λ对齐，λ通过单品销售量/品类销售量计算

# 认为成本加成定价系数是r_{i,t},研究目标是Aout_i和r_i,t之间的关系，最终还要对所有的r_i,t统一处理得到r_i
# 从file_2可以知道每个单品每天卖的量是多少也就是Aout，r = P/C -1
# 销售单价p 批发价c分别可以从file_2 file_3得知
# 先给出Aout和r之间的描述性统计，再考虑是否要进行回归
# 总体决定先做prophet模型，再考虑是否用LSTM预测

# 计算r_i,t
import pandas as pd
import numpy as np

# 定价系数r_i,t = P_i,t/C_i,t - 1
df = pd.read_csv('file_23.csv')
df['销售单价(元/千克)'] = df['销售单价(元/千克)'].astype(float)
df['批发价格(元/千克)'] = df['批发价格(元/千克)'].astype(float)
df['r_i,t'] = df['销售单价(元/千克)']/df['批发价格(元/千克)'] - 1
df.to_csv('file_23_r.csv', index=False,encoding = 'utf-8-sig')


# 将file_23_r和file_1合并，file_1是单品编码和分类名称的对应关系
df_1 = pd.read_csv('file_1.csv')
df_1['单品编码'] = df_1['单品编码'].astype(str)
df_1['分类名称'] = df_1['分类名称'].astype(str)

df_23_r = pd.read_csv('file_23_r.csv')
df_23_r['单品编码'] = df_23_r['单品编码'].astype(str)

# 合并file_23_r和file_1，获取分类名称信息
df_merged = pd.merge(df_23_r, df_1[['单品编码', '分类名称']], on='单品编码', how='left')

# 检查是否有未匹配的单品编码
unmatched = df_merged[df_merged['分类名称'].isna()]
if len(unmatched) > 0:
    print(f"警告：有 {len(unmatched)} 个单品编码未找到对应的分类名称")
    print("未匹配的单品编码：", unmatched['单品编码'].unique()[:10])  # 显示前10个

# 计算权重系数λ，λ_i,t = Aout_i,t/Aout_t 
# 其中 Aout_i,t 是单品i在时间t的销量，Aout_t 是该单品所属品类在时间t的总销量

# 为合并后的数据添加每日品类销量列（每个单品都能看到自己所属品类当天的总销量）
df_merged['每日品类销量'] = df_merged.groupby(['销售日期', '分类名称'])['销量(千克)'].transform('sum')

# 计算每个单品的权重系数λ
df_merged['lambda_i,t'] = df_merged['销量(千克)'] / df_merged['每日品类销量']

# 将同一天相同单品编码的λ值合并（相加）
df_merged_consolidated = df_merged.groupby(['销售日期', '单品编码', '分类名称']).agg({
    '销量(千克)': 'sum',  # 销量相加
    '销售单价(元/千克)': 'mean',  # 价格取平均
    '批发价格(元/千克)': 'mean',  # 批发价取平均
    'r_i,t': 'mean',  # 定价系数取平均
    '每日品类销量': 'first',  # 品类销量保持不变
    'lambda_i,t': 'sum'  # λ值相加
}).reset_index()

# 验证权重系数计算是否正确（同一品类同一天的λ值之和应该等于1）
validation = df_merged_consolidated.groupby(['销售日期', '分类名称'])['lambda_i,t'].sum().reset_index()
print("权重系数验证（同一品类同一天的λ值之和）：")
print(validation.head(10))

# 检查合并前后的数据行数
print(f"合并前数据行数：{len(df_merged)}")
print(f"合并后数据行数：{len(df_merged_consolidated)}")

# 保存结果
df_merged.to_csv('file_23_r_with_category.csv', index=False, encoding='utf-8-sig')
df_merged_consolidated.to_csv('file_23_r_with_category_consolidated.csv', index=False, encoding='utf-8-sig')
print(f"合并完成，共 {len(df_merged)} 行数据")
print(f"λ值合并完成，共 {len(df_merged_consolidated)} 行数据")
print(f"权重系数λ计算完成，已添加到合并数据中")
print(f"每日品类销量已添加到合并数据中")
print(f"合并后权重系数λ的范围：{df_merged_consolidated['lambda_i,t'].min():.4f} 到 {df_merged_consolidated['lambda_i,t'].max():.4f}")


